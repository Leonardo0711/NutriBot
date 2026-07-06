"""
Nutribot Backend - Interactive Message Factory
"""
from __future__ import annotations


def _scale_row(prefix: str, value: int, max_value: int) -> dict:
    return {
        "id": f"{prefix}:{value}",
        "title": str(value),
        "description": "",
    }


def build_yes_no_buttons(
    body: str,
    button_yes_id: str,
    button_no_id: str,
    yes_label: str = "Sí",
    no_label: str = "No",
) -> dict:
    return {
        "type": "list",
        "body": body,
        "title": "Opciones",
        "buttonText": "Seleccionar",
        "sections": [
            {
                "title": "Opciones",
                "rows": [
                    {"id": button_yes_id, "title": yes_label, "description": ""},
                    {"id": button_no_id, "title": no_label, "description": ""},
                ],
            }
        ],
    }


def build_scale_list(
    body: str,
    prefix: str,
    min_value: int,
    max_value: int,
    title: str = "Selecciona una opcion",
) -> dict:
    return {
        "type": "list",
        "body": body,
        "title": "Encuesta",
        "buttonText": "Elegir",
        "sections": [
            {
                "title": title,
                "rows": [
                    _scale_row(prefix, i, max_value)
                    for i in range(min_value, max_value + 1)
                ],
            }
        ],
    }
