from __future__ import annotations

from typing import Optional

from app.models.ib_prodclass import IB_PRODCLASS
from app.services.analyzer import build_prompt as build_main_prompt


def _sanitize(value: Optional[str]) -> str:
    if not value:
        return "—"
    payload = value.strip()
    return payload or "—"


def build_site_available_prompt(
    text_par: str,
    company_name: str,
    okved: str,
) -> str:
    """Построить промпт для случая, когда текст сайта доступен."""

    return build_main_prompt(text_par, company_name, okved)


def build_okved_prompt(okved: str) -> str:
    """Построить промпт для случая, когда доступен только ОКВЭД."""

    instructions = """
Напиши ответ в виде одной только цифры IB_PRODCLASS,
где
PRODCLASS_by_OKVED = ID класса производства ib_prodclass (список ниже), который ближе всего по сходству с ОКВЭД {OKVED};
"""

    table = "\n".join(f'{k}:"{IB_PRODCLASS[k]}"' for k in sorted(IB_PRODCLASS))

    return (
        "{instructions}\n"
        "IB_PRODCLASS = {{\n{table}\n}}"
    ).format(
        instructions=instructions.strip().replace("{OKVED}", _sanitize(okved)),
        table=table,
    )
