from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from app.config import settings


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


# Значения можно обновлять по мере изменений прайса OpenAI.
# Источник: официальная pricing-страница OpenAI (Standard + Embeddings).
_STANDARD_MODEL_PRICING_USD_PER_1M: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_per_1m=5.00, cached_input_per_1m=2.50, output_per_1m=15.00),
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, cached_input_per_1m=0.075, output_per_1m=0.60),
    "gpt-5-mini": ModelPricing(input_per_1m=0.45, cached_input_per_1m=0.045, output_per_1m=3.60),
    "gpt-5.2": ModelPricing(input_per_1m=1.25, cached_input_per_1m=0.125, output_per_1m=10.00),
    "text-embedding-3-small": ModelPricing(input_per_1m=0.02, cached_input_per_1m=0.0, output_per_1m=0.0),
    "text-embedding-3-large": ModelPricing(input_per_1m=0.13, cached_input_per_1m=0.0, output_per_1m=0.0),
}

# Храним таблицы отдельно по tier. Для неподдерживаемых/отсутствующих вариаций
# используются значения Standard.
MODEL_PRICING_USD_PER_1M_BY_TIER: Dict[str, Dict[str, ModelPricing]] = {
    "STANDARD": dict(_STANDARD_MODEL_PRICING_USD_PER_1M),
    "PRIORITY": dict(_STANDARD_MODEL_PRICING_USD_PER_1M),
    "BATCH": dict(_STANDARD_MODEL_PRICING_USD_PER_1M),
    "FLEX": dict(_STANDARD_MODEL_PRICING_USD_PER_1M),
}


def _safe_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _resolve_tier_pricing() -> Mapping[str, ModelPricing]:
    tier = (settings.OPENAI_PRICING_TIER or "STANDARD").strip().upper() or "STANDARD"
    return MODEL_PRICING_USD_PER_1M_BY_TIER.get(tier, MODEL_PRICING_USD_PER_1M_BY_TIER["STANDARD"])


def calculate_response_cost_usd(model: str, usage: Dict[str, Any]) -> float:
    pricing = _resolve_tier_pricing().get((model or "").strip())
    if pricing is None:
        return 0.0

    # Для embeddings API обычно приходит total_tokens.
    input_tokens = _safe_int(usage.get("input_tokens"))
    if input_tokens <= 0:
        input_tokens = _safe_int(usage.get("total_tokens"))

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
