"""Optional GPT memo layer (numbers are injected verbatim from simulator output)."""

from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover — optional dependency for local dev
    OpenAI = None


MEMO_SCHEMA = (
    '{"headline":"<string>","recommend":"<scenario_a_name|scenario_b_name|neutral>",'
    '"bullets":["<string>","<string>","<string>"],'
    '"risks":["<string>","<string>"],'
    '"confidence_note":"<short string acknowledging assumptions>"}'
)


def synthesize_memo(metrics_context: dict[str, Any], api_key: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    client = OpenAI(api_key=api_key)
    payload = json.dumps(metrics_context, indent=2)

    system = (
        "You summarize retention decision simulations for executives. "
        "You MUST only restate quantitative facts supplied in CONTEXT JSON "
        "(medians, p05/p95, prob_negative, intervention names, cohort counts). "
        "Do NOT invent new statistics, uplift claims, causal guarantees, or customer examples. "
        f"Respond with valid JSON shaped like: {MEMO_SCHEMA}. "
        "The recommend field must match one of the scenario names in CONTEXT exactly, "
        "or the literal string neutral for a tie or unclear tradeoff. "
    )
    user = f"CONTEXT (authoritative):\n```json\n{payload}\n```\nProduce the JSON."

    rsp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )

    txt = rsp.choices[0].message.content or "{}"
    return json.loads(txt)
