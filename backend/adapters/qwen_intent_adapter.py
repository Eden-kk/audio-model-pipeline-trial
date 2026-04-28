"""Qwen-based intent_llm adapter for the slow-loop pipeline.

Reads stage 2 (ASR text) + stage 3 (per-segment user-tag) and produces the
slow-loop envelope:

    {
      memory_doc:    short Markdown summary of wearer-only utterances,
      tool_calls:    [{name, args}, ...],
      salient_facts: [str, ...],
    }

Wraps any OpenAI-compatible chat-completions endpoint via env
INTENT_LLM_URL (e.g. the existing exp06-qwen25-openai-server Modal
deployment, OpenAI itself, or any vLLM-served Qwen) — pick the model
via env INTENT_LLM_MODEL.

Env:
    INTENT_LLM_URL    base URL with /v1 (e.g. https://...modal.run/v1)
    INTENT_LLM_MODEL  default 'Qwen/Qwen2.5-7B-Instruct'
    INTENT_LLM_KEY    optional API key (omit → no auth header)
"""
from __future__ import annotations

import json as _json
import os
import time
from typing import Any, Dict, List, Optional

import httpx


_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_TOKENS = 800

_SYSTEM_PROMPT = """\
You are the slow-loop memory + tool-call extractor for an AR-glasses voice
assistant. Your input is a 30-second window of transcribed audio with
per-segment speaker tags. Only the WEARER's utterances may produce tool
calls. Other speakers' utterances are ambient context.

Return a SINGLE JSON object with exactly these keys (no markdown, no prose):

  {
    "memory_doc": str,         // 1-3 sentence Markdown summary of wearer turns + ambient context.
                               // Empty string if no wearer speech in this window.
    "tool_calls": [            // empty list if nothing actionable
      {"name": str, "args": object}
    ],
    "salient_facts": [str]     // canonical short facts the assistant should
                               // remember long-term. Empty if none.
  }

Tool-call shape:
  - {"name": "execute", "args": {"task": "<user-intent in one sentence>"}}
  - {"name": "remember", "args": {"fact": "<canonical short fact>"}}

Rules:
  1. Tool calls fire ONLY for wearer utterances (is_user=true segments).
  2. If no wearer utterance present, return tool_calls=[] and salient_facts=[].
  3. memory_doc must be plain text — no markdown headings, no JSON inside.
  4. Output the JSON object only — no preamble, no trailing text.
"""


class QwenIntentAdapter:
    id = "qwen_intent"
    category = "intent_llm"
    display_name = "Qwen-Intent (OpenAI-compat)"
    hosting = "modal"
    vendor = "Qwen via vLLM"
    is_streaming = False

    inputs: List[Dict[str, str]] = [
        {"name": "transcript", "type": "text"},
        {"name": "speaker_segments", "type": "speaker_segments"},
    ]
    outputs: List[Dict[str, str]] = [
        {"name": "envelope", "type": "memory_doc"},
        {"name": "tool_calls", "type": "tool_calls"},
    ]
    config_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "model": {"type": "string", "default": _DEFAULT_MODEL,
                      "description": "Override the OpenAI-compat model name."},
            "temperature": {"type": "number", "default": _DEFAULT_TEMPERATURE},
            "max_tokens": {"type": "integer", "default": _DEFAULT_MAX_TOKENS},
        },
    }
    cost_per_call_estimate_usd: Optional[float] = None

    def _url(self) -> str:
        url = os.environ.get("INTENT_LLM_URL", "")
        if not url:
            raise RuntimeError(
                "INTENT_LLM_URL not set — point this at your OpenAI-compat /v1 "
                "endpoint (e.g. exp06-qwen25-openai-server's Modal URL)."
            )
        return url.rstrip("/")

    def _key(self) -> Optional[str]:
        return os.environ.get("INTENT_LLM_KEY") or None

    def _build_user_payload(self, payload: dict) -> str:
        """Compose the user-message body the LLM sees from the runner's
        stage-input dict."""
        text = payload.get("text", "")
        words = payload.get("words") or []
        speaker_segments = payload.get("speaker_segments") or []
        language = payload.get("language", "en")

        # Project speaker_segments → compact list the LLM can read
        seg_lines = []
        for s in speaker_segments:
            seg_lines.append(
                f"  [{s.get('start', 0):.1f}-{s.get('end', 0):.1f}s] "
                f"{'WEARER' if s.get('is_user') else 'OTHER'} "
                f"(score={s.get('score', 0):.2f})"
            )
        seg_block = "\n".join(seg_lines) if seg_lines else "  (no per-segment tags available)"

        # If word-level speaker info is present, show wearer-only words
        wearer_words = [w.get("word", "") for w in words if w.get("speaker") == "wearer"]
        wearer_excerpt = " ".join(wearer_words)[:500] if wearer_words else ""

        body = (
            f"Language: {language}\n\n"
            f"Per-segment speaker tags (1-second sliding window):\n"
            f"{seg_block}\n\n"
            f"Full transcript:\n{text}\n"
        )
        if wearer_excerpt:
            body += f"\nWord-level wearer-only excerpt:\n{wearer_excerpt}\n"
        return body

    async def infer(self, payload: dict, config: dict) -> dict:
        model = config.get("model", _DEFAULT_MODEL)
        temperature = float(config.get("temperature", _DEFAULT_TEMPERATURE))
        max_tokens = int(config.get("max_tokens", _DEFAULT_MAX_TOKENS))

        user_body = self._build_user_payload(payload)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_body},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{self._url()}/chat/completions",
                                  json=body, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"intent_llm HTTP {r.status_code}: {r.text[:300]}")
        wall_s = time.perf_counter() - t0

        resp = r.json()
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = resp.get("usage") or {}

        # Parse the JSON envelope; fall back to {} on malformed output so the
        # runner doesn't crash — the dispatch stage will write whatever we
        # got and log the failure.
        envelope: Dict[str, Any] = {}
        try:
            envelope = _json.loads(content)
        except _json.JSONDecodeError:
            envelope = {"memory_doc": content[:500], "tool_calls": [],
                        "salient_facts": [], "_parse_error": True}

        # Normalise fields the downstream dispatch stage relies on.
        envelope.setdefault("memory_doc", "")
        envelope.setdefault("tool_calls", [])
        envelope.setdefault("salient_facts", [])

        return {
            **envelope,
            "model": model,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "wall_time_s": wall_s,
            "cost_usd": 0.0,    # self-hosted Qwen — actual $ tracked at infra layer
            "raw_response": resp,
        }
