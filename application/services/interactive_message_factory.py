"""
Nutribot Backend - Interactive Message Factory
"""
from __future__ import annotations


def build_yes_no_buttons(
    body: str,
    button_yes_id: str,
    button_no_id: str,
    yes_label: str = "Si",
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
                    {
                        "id": f"{prefix}:{i}",
                        "title": str(i),
                        "description": "",
                    }
                    for i in range(min_value, max_value + 1)
                ],
            }
        ],
    }
