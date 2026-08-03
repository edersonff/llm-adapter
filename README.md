# llm

One binary in front of every LLM provider. Same request shape in, same response shape out, no matter
which provider actually answers. Routes by latency, rotates keys that hit a rate limit, falls back to
another model when one is down, streams when asked.

## Input

JSON on stdin, an OpenAI-style `messages` array:

```json
{
  "model": "glm-5-turbo",
  "messages": [{ "role": "user", "content": "hello" }],
  "temperature": 0.7,
  "max_tokens": 4096
}
```

Flags: `--config` (default `config.yaml`), `--model` (overrides the body), `--stream`, `--verbose`.

## Output

Non-streaming: one OpenAI-style completion object on stdout.

```json
{ "choices": [{ "message": { "content": "..." }, "finish_reason": "stop" }],
  "usage": { "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30 } }
```

Streaming (`--stream`): SSE lines on stdout, `data: {...}`, ending with `data: [DONE]`. Some models
send `reasoning_content` chunks before the real answer — that is the model thinking out loud, not a
bug, and callers that only want the final text should skip deltas that carry it.

Everything else — routing decisions, retries, key rotation — goes to stderr, never stdout.

## Run it

    cargo build --release
    echo '{"messages":[{"role":"user","content":"hello"}]}' | target/release/llm-adapter --config config.yaml

## What it needs

Rust to build it once. Nothing to install to run it after that. Network access to whichever provider
`config.yaml` points at.

## Settings, and what they default to

Everything lives in `config.yaml`, copied from `config.yaml.template` and filled with real keys —
never commit the filled one.

    providers      base url and a list of keys per provider, rotated on failure
    models         which provider serves each model name, token limits, timeout
    routing        latency-based by default, 3 fails before a key cools down 120s
    fallbacks      ordered list of models to try if the first one fails

## What breaks

Measured against the real provider, not guessed:

- Malformed JSON on stdin: exit 1, `failed to parse stdin JSON: ...` with the parser's line and
  column. That message comes from the wrapped binary, not from sheol, and it is the one rough edge
  here — still useful, just terser than the rest of this module's errors.
- Missing config file: exit 1, and the message includes a raw `os error 2`. Same origin, same caveat.
- A key that is dead or over quota: the binary rotates to the next one automatically. You only see a
  failure if every key in the list is out.
- `exit 0` success. `exit 1` config or validation error. `exit 2` API, network or stream error.

### Known issue: asyncio subprocess deadlock

Calling this binary from Python with `asyncio.create_subprocess_exec` + `communicate()` can hang under
concurrent calls — a real Python bug ([gh-146181](https://github.com/python/cpython/issues/146181)),
not this module. `create_subprocess_exec` does blocking I/O on the event loop thread, and concurrent
spawns stall it. Fix: run the sync `subprocess.Popen` inside `asyncio.to_thread()` instead. See
`adapter_client.py` for the client that already does this.

Related: [gh-141473](https://github.com/python/cpython/issues/141473) (`communicate()` drops input
after a timeout, fixed in 3.14.1), [agentscope#1255](https://github.com/agentscope-ai/agentscope/issues/1255)
(deadlock when output exceeds the pipe buffer). Open as of 2026-04, affects Python 3.x generally.
