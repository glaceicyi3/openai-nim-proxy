// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased payload limit
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to false to hide thinking (recommended for GLM)

// 🔥 REASONING EFFORT - GLM-5.3 defaults to 'max' if unset, which can mean
// a very long internal reasoning pass before any answer is emitted. Options
// per NVIDIA's model card: 'low', 'high', 'max'. Use 'low' for snappy chat/
// roleplay responses; bump to 'high' if answer quality suffers.
const GLM_REASONING_EFFORT = 'low';

// 🔥 DEBUG TOGGLE - Logs every raw SSE chunk received from NIM. Turn this on
// temporarily if a stream dies partway through, to see exactly where/how it
// stops (silent socket death vs. a malformed/unexpected chunk).
const DEBUG_RAW_CHUNKS = false;

// 🔥 STREAM IDLE TIMEOUT (ms) - If no data arrives from NIM for this long
// mid-stream, we abort cleanly instead of hanging forever. Long reasoning
// traces can have real gaps, so keep this generous but finite.
const STREAM_IDLE_TIMEOUT_MS = 90000; // 90s of silence = treat as dead

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'z-ai/glm-5.3',
  'gpt-4': 'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-8b-instruct',
  'claude-3-opus': 'meta/llama-3.3-70b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'deepseek-ai/deepseek-v3.1'
};

// Match any GLM model by prefix instead of hardcoding exact version
// strings everywhere. Update MODEL_MAPPING above when NVIDIA renames an
// endpoint; this check then keeps working without further edits.
function isGLM(nimModel) {
  return typeof nimModel === 'string' && nimModel.toLowerCase().startsWith('z-ai/glm');
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    glm_reasoning_effort: GLM_REASONING_EFFORT,
    debug_raw_chunks: DEBUG_RAW_CHUNKS
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const requestStartTime = Date.now();

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Transform OpenAI request to NIM format
    let processedMessages = messages;

    // Add natural writing instruction for GLM models
    if (isGLM(nimModel)) {
      // Check if there's already a system message
      const hasSystemMessage = messages.some(msg => msg.role === 'system');

      if (!hasSystemMessage) {
        processedMessages = [
          {
            role: 'system',
            content: `You are an expert creative writer specializing in immersive, natural roleplay and narrative storytelling. Follow these guidelines strictly:

THINKING PROCESS (Internal - not shown to user):
- Think in natural, flowing narrative style
- Analyze character motivations, subtext, and emotional dynamics naturally
- Consider: "This moment reveals X about the character... The tension here stems from Y..."
- DO NOT use structured formats like "Let's analyze:", "User's response:", "Example dialogues:"
- Think like you're crafting a story, not filling out a template

WRITING STYLE:
- Write in flowing, descriptive prose with varied sentence structures
- Combine related actions and thoughts into complex, natural sentences
- NEVER use choppy fragments like "He did this. Did that. Then this."
- Instead write: "He took a deep breath before opening his eyes, his thoughts fuzzy and scattered."

PARAGRAPH STRUCTURE:
- Separate distinct moments, scenes, or shifts in focus into paragraphs
- Each paragraph should contain 4-6 complete sentences that flow together
- Use proper line breaks (double newlines) between paragraphs
- Never create walls of text - always break into digestible paragraphs

FORMATTING:
- Use proper markdown: *italics* NOT * italics * (no spaces inside asterisks)
- Italicize internal thoughts, emphasis, and foreign words correctly
- Maintain consistent formatting throughout the response
- Ensure all opening markdown tags are properly closed

DESCRIPTION & DEPTH:
- Layer physical actions with internal thoughts and emotional reactions
- Use sensory details and atmospheric description naturally within the narrative
- Build tension and pacing through sentence rhythm and structure
- Show character depth through reaction, not just action

AVOID:
- Staccato, telegram-style sentences
- Repetitive sentence patterns
- Inconsistent or broken markdown formatting
- Single-sentence paragraphs (unless for dramatic effect)
- Walls of text without paragraph breaks
- Structured, bullet-point thinking (analyze the scene naturally)

Write as if you are crafting a published novel - polished, immersive, and engaging.`
          },
          ...messages
        ];
      }
    }

    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.8,
      top_p: 0.95,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };

    // Add thinking parameters for GLM models.
    // - reasoning_effort is a top-level request field (like OpenAI's
    //   o-series models) - it defaults to 'max' if omitted, which is the
    //   main cause of very long delays before any output appears.
    // - clear_thinking lives inside chat_template_kwargs and defaults to
    //   false; for multi-turn chat that means every turn's reasoning trace
    //   stays in context and compounds, slowing each subsequent turn down.
    //   Chat/roleplay use cases should set this true.
    if (isGLM(nimModel)) {
      nimRequest.reasoning_effort = GLM_REASONING_EFFORT;
      nimRequest.chat_template_kwargs = {
        clear_thinking: true
      };
      console.log(`[GLM] Request to ${nimModel} with reasoning_effort=${GLM_REASONING_EFFORT}, clear_thinking=true`);
    } else {
      // Non-GLM models: no reasoning params to add.
    }

    // Log which model is being used
    console.log(`[REQUEST] Using NVIDIA model: ${nimModel}`);

    // Make request to NVIDIA NIM API with retry logic for overloaded models
    let response;
    let lastError;
    const maxRetries = 3;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`[ATTEMPT ${attempt}/${maxRetries}] Calling NVIDIA API for ${nimModel}`);

        response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 300000 // 5 minutes timeout for large models like GLM-5.3
        });

        // Success! Break out of retry loop
        console.log(`[SUCCESS] Got response from ${nimModel}`);
        break;

      } catch (error) {
        lastError = error;
        console.error('Proxy error:', error.message);

        // If it's a 404, the model truly doesn't exist - don't retry
        if (error.response?.status === 404) {
          console.error(`[404] Model ${nimModel} not found - not retrying`);
          throw error;
        }

        // If it's 429 (rate limit) or 503 (service unavailable), retry with backoff
        if (error.response?.status === 429 || error.response?.status === 503 || error.code === 'ECONNABORTED') {
          const waitTime = attempt * 2000; // 2s, 4s, 6s backoff
          console.log(`[RETRY] ${nimModel} overloaded (${error.response?.status || error.code}), waiting ${waitTime}ms before retry ${attempt}/${maxRetries}`);

          if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, waitTime));
            continue;
          }
        }

        // Other errors - throw immediately
        throw error;
      }
    }

    // If we exhausted all retries, throw the last error
    if (!response) {
      throw lastError;
    }

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      // Running state for GLM paragraph-break formatting, kept O(1) per
      // chunk instead of re-scanning the whole accumulated response.
      let glmSentenceCount = 0;
      let glmTailBuffer = ''; // small rolling tail, not the full response

      // --- Idle-timeout watchdog -------------------------------------
      // If NIM goes silent mid-stream (common with long GLM reasoning
      // traces on an overloaded endpoint), the underlying socket can
      // die without ever firing 'error' or 'end'. Without this, the
      // client just hangs forever. We reset the timer on every chunk
      // and abort cleanly if it fires.
      let idleTimer = null;
      let finished = false;

      const clearIdleTimer = () => {
        if (idleTimer) {
          clearTimeout(idleTimer);
          idleTimer = null;
        }
      };

      const armIdleTimer = () => {
        clearIdleTimer();
        idleTimer = setTimeout(() => {
          if (finished) return;
          console.error(`[IDLE TIMEOUT] No data from ${nimModel} for ${STREAM_IDLE_TIMEOUT_MS}ms - aborting stream`);
          finished = true;
          try {
            // Let the client know generation was cut off, then close.
            res.write(`data: ${JSON.stringify({
              error: { message: 'Upstream stream stalled and was aborted by proxy', code: 'idle_timeout' }
            })}\n\n`);
            res.write('data: [DONE]\n\n');
          } catch (e) {
            // response may already be closed
          }
          res.end();
          if (response.data && typeof response.data.destroy === 'function') {
            response.data.destroy();
          }
        }, STREAM_IDLE_TIMEOUT_MS);
      };

      armIdleTimer();

      // Timing instrumentation: logs the gap since the previous chunk and
      // the total elapsed time since the request started, so a single long
      // pause vs. a continuous slow trickle can be told apart from the logs.
      let lastChunkAt = requestStartTime;
      let chunkCount = 0;
      let firstByteLogged = false;

      response.data.on('data', (chunk) => {
        if (finished) return;
        armIdleTimer(); // saw data, push the deadline back out

        const now = Date.now();
        chunkCount++;
        if (!firstByteLogged) {
          console.log(`[TIMING] First byte from ${nimModel} after ${now - requestStartTime}ms`);
          firstByteLogged = true;
        }
        const gap = now - lastChunkAt;
        if (gap > 2000) {
          console.log(`[TIMING] Chunk #${chunkCount} arrived after a ${gap}ms gap (total elapsed ${now - requestStartTime}ms)`);
        }
        lastChunkAt = now;

        if (DEBUG_RAW_CHUNKS) {
          console.log('[RAW CHUNK]', chunk.toString().slice(0, 300));
        }

        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              // SSE frames must end on a blank line to be recognized as
              // closed by strict clients - use \n\n, not \n.
              res.write('data: [DONE]\n\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                let finalContent = '';

                if (SHOW_REASONING) {
                  let combinedContent = '';

                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  finalContent = combinedContent;
                } else {
                  finalContent = content || '';
                  // Don't include reasoning in output
                }

                // For GLM models, add extra line breaks after sentence-ending
                // punctuation. Fixed to only look at a small rolling tail
                // instead of re-matching the entire accumulated response on
                // every chunk (that was O(n^2) and would visibly slow the
                // stream down the longer a response ran).
                if (isGLM(nimModel) && finalContent) {
                  glmTailBuffer = (glmTailBuffer + finalContent).slice(-200);

                  const sentenceEnders = (finalContent.match(/[.!?](?:\s*["']?\s+[A-Z]|$)/g) || []).length;
                  if (sentenceEnders > 0) {
                    const before = glmSentenceCount;
                    glmSentenceCount += sentenceEnders;

                    const crossedFourBoundary = Math.floor(glmSentenceCount / 4) > Math.floor(before / 4);
                    if (crossedFourBoundary && /[.!?]\s*["']?\s*$/.test(finalContent)) {
                      finalContent = finalContent + '\n\n';
                    }
                  }
                }

                if (finalContent) {
                  data.choices[0].delta.content = finalContent;
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              if (DEBUG_RAW_CHUNKS) {
                console.error('[PARSE ERROR]', e.message, 'line:', line.slice(0, 300));
              }
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => {
        if (finished) return;
        finished = true;
        clearIdleTimer();
        console.log(`[TIMING] Stream from ${nimModel} finished after ${Date.now() - requestStartTime}ms total, ${chunkCount} chunks`);
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        if (finished) return;
        finished = true;
        clearIdleTimer();
        res.end();
      });

      // If the client disconnects, stop the upstream request too.
      req.on('close', () => {
        if (finished) return;
        finished = true;
        clearIdleTimer();
        if (response.data && typeof response.data.destroy === 'function') {
          response.data.destroy();
        }
      });

    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          // Paragraph formatting for GLM models
          let formattedContent = fullContent;
          if (isGLM(nimModel)) {
            // First, fix broken markdown formatting (spaces inside asterisks)
            formattedContent = formattedContent.replace(/\*\s+/g, '*').replace(/\s+\*/g, '*');

            // Split text into sentences (improved regex for dialogue and complex punctuation)
            const sentences = formattedContent.match(/[^.!?]+[.!?]+["']?(?=\s+[A-Z]|$)/g) || [formattedContent];

            // Group sentences into paragraphs of ~5 sentences
            const paragraphs = [];
            for (let i = 0; i < sentences.length; i += 5) {
              const paragraph = sentences.slice(i, i + 5).join(' ').trim();
              if (paragraph.length > 0) {
                paragraphs.push(paragraph);
              }
            }

            // Join with double newlines and clean up excessive spacing
            formattedContent = paragraphs.join('\n\n').replace(/\n{3,}/g, '\n\n');
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: formattedContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`GLM reasoning effort: ${GLM_REASONING_EFFORT}`);
  console.log(`Stream idle timeout: ${STREAM_IDLE_TIMEOUT_MS}ms`);
});
