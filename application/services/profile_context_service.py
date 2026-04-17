"""
Nutribot Backend - Profile Context Service
"""
from __future__ import annotations

from domain.profile_snapshot import ProfileSnapshot


class ProfileContextService:
    def _fmt(self, val, unit: str = "") -> str:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return "Pendiente"
        return f"{val}{unit}"

    def _fmt_height(self, val_cm) -> str:
        if val_cm is None:
            return "Pendiente"
        try:
            val = float(val_cm)
            if val > 10:
                return f"{val / 100:.2f}m"
            return f"{val:.2f}m"
        except Exception:
            return str(val_cm)

    def _fmt_list(self, values: tuple[str, ...]) -> str:
        return ", ".join(values) if values else "Ninguna"

    def human_step_label(self, step_code: str) -> str:
        labels = {
            "edad": "edad",
            "peso_kg": "peso",
            "altura_cm": "talla",
            "alergias": "alergias",
            "enfermedades": "condiciones de salud",
            "restricciones_alimentarias": "restricciones alimentarias",
            "tipo_dieta": "tipo de dieta",
            "objetivo_nutricional": "objetivo nutricional",
            "provincia": "provincia",
            "distrito": "distrito",
        }
        return labels.get(step_code, step_code)

    def pending_fields(self, snapshot: ProfileSnapshot) -> list[str]:
        checks = [
            ("edad", "edad"),
            ("peso_kg", "peso"),
            ("altura_cm", "talla"),
            ("alergias", "alergias"),
            ("enfermedades", "condiciones de salud"),
            ("restricciones_alimentarias", "restricciones"),
            ("tipo_dieta", "tipo de dieta"),
            ("objetivo_nutricional", "objetivo"),
            ("provincia", "provincia"),
            ("distrito", "distrito"),
        ]
        pending = []
        for step_code, label in checks:
            if step_code in snapshot.skipped_fields:
                continue
            value = snapshot.value_for_step(step_code)
            if value is None or (isinstance(value, str) and not value.strip()):
                pending.append(label)
        return pending

    def missing_essential_fields(self, snapshot: ProfileSnapshot) -> list[str]:
        essentials = ["edad", "peso_kg", "altura_cm", "alergias"]
        missing = []
        for step_code in essentials:
            if step_code in snapshot.skipped_fields:
                continue
            value = snapshot.value_for_step(step_code)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(step_code)
        return missing

    def build_prompt_and_summary(self, snapshot: ProfileSnapshot) -> tuple[str, str]:
        parts = [
            f"Edad: {self._fmt(snapshot.measurements.age_years, ' años')}",
            f"Peso: {self._fmt(snapshot.measurements.weight_kg, 'kg')}",
            f"Talla: {self._fmt_height(snapshot.measurements.height_cm)}",
            f"Tipo de dieta: {self._fmt(snapshot.health.diet_type)}",
            f"Alergias: {self._fmt_list(snapshot.health.allergies)}",
            f"Enfermedades: {self._fmt_list(snapshot.health.diseases)}",
            f"Restricciones: {self._fmt_list(snapshot.health.food_restrictions)}",
            f"Objetivo: {self._fmt(snapshot.health.nutrition_goal)}",
            f"Ubicación: {self._fmt(snapshot.location.best_location_text)}",
        ]
        profile_text = "\n[DATOS ACTUALES DEL PERFIL DEL USUARIO]\n- " + "\n- ".join(parts)
        summary = (
            f"• Edad: {self._fmt(snapshot.measurements.age_years, ' años')}\n"
            f"• Peso: {self._fmt(snapshot.measurements.weight_kg, 'kg')}\n"
            f"• Talla: {self._fmt_height(snapshot.measurements.height_cm)}\n"
            f"• Alergias: {self._fmt_list(snapshot.health.allergies)}\n"
            f"• Enfermedades: {self._fmt_list(snapshot.health.diseases)}\n"
            f"• Objetivo: {self._fmt(snapshot.health.nutrition_goal)}"
        )
        return profile_text, summary

    def recommendation_citation(self, snapshot: ProfileSnapshot) -> str:
        citation = "Considerando"
        age = snapshot.measurements.age_years
        weight = snapshot.measurements.weight_kg
        height = snapshot.measurements.height_cm
        if age is not None or weight is not None or height is not None:
            age_txt = age if age is not None else "?"
            weight_txt = weight if weight is not None else "?"
            h_str = self._fmt_height(height) if height is not None else "?"
            citation += f" que tienes {age_txt} años, pesas {weight_txt}kg y mides {h_str}"
            if snapshot.health.allergies:
                citation += f", tienes alergia a {self._fmt_list(snapshot.health.allergies)}"
            if snapshot.health.nutrition_goal:
                citation += f" y tu objetivo es {snapshot.health.nutrition_goal}"
        else:
            citation += " tus datos actuales"
        return citation + ":"
