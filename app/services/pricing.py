from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


# Значения можно обновлять по мере изменений прайса OpenAI.
MODEL_PRICING_USD_PER_1M: Dict[str, ModelPricing] = {
    "gpt-5-mini": ModelPricing(input_per_1m=0.45, cached_input_per_1m=0.045, output_per_1m=3.60),
    "gpt-5.2": ModelPricing(input_per_1m=1.25, cached_input_per_1m=0.125, output_per_1m=10.0),
}


def _safe_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def calculate_response_cost_usd(model: str, usage: Dict[str, Any]) -> float:
    pricing = MODEL_PRICING_USD_PER_1M.get((model or "").strip())
    if pricing is None:
        return 0.0

    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))

    details = usage.get("input_tokens_details")
    cached_tokens = 0
    if isinstance(details, dict):
        cached_tokens = _safe_int(details.get("cached_tokens"))

    uncached_input = max(input_tokens - cached_tokens, 0)

    return (
        uncached_input * (pricing.input_per_1m / 1_000_000)
        + cached_tokens * (pricing.cached_input_per_1m / 1_000_000)
        + output_tokens * (pricing.output_per_1m / 1_000_000)
    )
