from __future__ import annotations

from typing import Optional

from app.models.ib_prodclass import IB_PRODCLASS
from app.services.analyzer import ib_prodclass_table_str


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

    rules_template = """
Ниже представлен текст с сайта компании.
Напиши ответ в таком виде:
[DESCRIPTION] =[описание компании по сайту]
[DESCRIPTION_SCORE] =[уровень соответствия содержимого сайта названием компании]
[OKVED_SCORE] =[уровень соответствия содержимого сайта ОКВЭД компании]
[PRODCLASS]=[ID класса производства ib_prodclass по содержимому сайта]
[PRODCLASS_SCORE]=[уровень сходства с ib_prodclass по содержимому сайта]
[EQUIPMENT_SITE]=[Оборудование1; Оборудование2; Оборудование3; и так далее]
[GOODS]=[Товар/услуга1; Товар/услуга2; Товар/услуга3; и так далее]
[GOODS_TYPE]=[ИмяТНВЭД1; ИмяТНВЭД2; ИмяТНВЭД3; и так далее]
, где
DESCRIPTION = описание компании по сайту
DESCRIPTION_SCORE = уровень сходства текста сайта компании с названием компании {Company_name}; число с точкой в диапазоне 0.00–1.00;
OKVED_SCORE = уровень сходства текста сайта компании с ОКВЭД компании {OKVED}; число с точкой в диапазоне 0.00–1.00;
PRODCLASS = ID класса производства ib_prodclass(список ниже), который ближе всего по сходству с текстом сайта;
PRODCLASS_SCORE= уровень смысловой связи (сходства) текста сайта компании с выбранным PRODCLASS; число с точкой в диапазоне 0.00–1.00;
EQUIPMENT_SITE = Список детальных наименований производственного/технологического оборудования, который упоминается на сайте и с помощью которого компания производит товары и или выполняет услуги, но не которое продает или реализует;
Пропускай наименования, которые слишком обширные, к примеру: «Промышленное оборудование», «Современная технологическая линия», "Высокотехнологические линии", "Современное оборудование" или близкие по смыслу. Либо добаляй к названию технологический процесс.
GOODS = Список товаров/услуг компании, который упоминается на сайте, которые она сама производит или реализует;
GOODS_TYPE = Для каждого GOODS нужно присвоить свой ТНВЭД и указать наименование ТНВЭД без кода.
Требования: текст и набор наименований для каждого параметра должен быть заключен в квадратные скобки [].
Название переменной и набор данных пишется с новой строки.
Отвечай строго без пояснений.
"""

    rules = (
        rules_template.replace("{Company_name}", _sanitize(company_name))
        .replace("{OKVED}", _sanitize(okved))
    )

    return (
        "{rules}\n"
        "Таблица ib_prodclass\n{table}\n"
        "Текст с сайта компании: {text}"
    ).format(rules=rules, table=ib_prodclass_table_str(), text=text_par)


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
