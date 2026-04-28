"""Pre-built pipeline recipes — the P0 vocabulary the user picks from in the
Pipelines page.  Each recipe declares an ordered list of adapter ids and
the runner walks them in sequence, threading the prior stage's output into
the next stage's input.

Today's recipes (all single-stage, since multi-stage adapter chaining
needs an output→input port-typing layer that lands in Slice 8):

  - asr-only:               clip → ASR adapter → text
  - tts-only:               text → TTS adapter → audio
  - speaker-verify-only:    clip + enrolled embedding → score
  - asr-then-tts:           clip → ASR → text → TTS → audio (round-trip
                            quality check; useful for noise-robustness)

A "recipe" is a Pipeline where `is_recipe=True`. The composer (Slice 8) lets
the user build their own pipelines on top of the same Pipeline schema.
"""
from __future__ import annotations

from typing import Any, Dict, List


def list_recipes() -> List[Dict[str, Any]]:
    """Return all built-in recipes, ready for /api/recipes to ship to the UI.

    Each recipe is a Pipeline-shaped dict: id, name, description, is_recipe,
    stages[], edges[].  The runner reads `stages` left-to-right and threads
    each stage's output into the next.  Stage ids match adapter category names
    so the UI can render category badges without an extra lookup.
    """
    return [
        {
            "id": "asr-only",
            "name": "ASR only",
            "description": (
                "Run a single ASR adapter on the clip's canonical audio and "
                "return text + word-level timing. Fastest single-stage recipe."
            ),
            "is_recipe": True,
            "stages": [
                {
                    "id": "asr",
                    "category": "asr",
                    "adapter": None,  # filled in at run time from the request
                    "config": {},
                },
            ],
            "edges": [],
        },
        {
            "id": "tts-only",
            "name": "TTS only",
            "description": (
                "Render text to audio with a TTS adapter. Measures both "
                "first-byte latency (TTFA) and full-render latency."
            ),
            "is_recipe": True,
            "stages": [
                {
                    "id": "tts",
                    "category": "tts",
                    "adapter": None,
                    "config": {},
                },
            ],
            "edges": [],
        },
        {
            "id": "speaker-verify-only",
            "name": "Speaker verify",
            "description": (
                "Score the clip against a previously-enrolled embedding. "
                "Pre-enroll a reference clip via /api/enroll first."
            ),
            "is_recipe": True,
            "stages": [
                {
                    "id": "speaker_verify",
                    "category": "speaker_verify",
                    "adapter": None,
                    "config": {},
                },
            ],
            "edges": [],
        },
        {
            "id": "asr-then-tts",
            "name": "ASR → TTS round-trip",
            "description": (
                "Two-stage: ASR transcribes the clip, then a TTS adapter "
                "speaks the transcript back. Useful as a quality probe — "
                "if the round-trip text matches the original speech, both "
                "adapters held up. Stages run sequentially; stage drill-in "
                "shows per-stage I/O + latency."
            ),
            "is_recipe": True,
            "stages": [
                {
                    "id": "asr",
                    "category": "asr",
                    "adapter": None,
                    "config": {},
                },
                {
                    "id": "tts",
                    "category": "tts",
                    "adapter": None,
                    "config": {},
                },
            ],
            "edges": [
                {"from": "asr", "to": "tts", "port": "text"},
            ],
        },
    ]


def get_recipe(recipe_id: str) -> Dict[str, Any] | None:
    for r in list_recipes():
        if r["id"] == recipe_id:
            return r
    return None
