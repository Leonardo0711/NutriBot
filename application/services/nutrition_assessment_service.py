"""
Nutribot Backend - NutritionAssessmentService
Cálculo e interpretación referencial del IMC.
No es diagnóstico médico, es orientación referencial.
"""
from __future__ import annotations

from typing import Optional

from domain.profile_snapshot import ProfileSnapshot


class NutritionAssessmentService:

    @staticmethod
    def compute_bmi(weight_kg: float, height_cm: float) -> Optional[float]:
        """Calcula el IMC. Retorna None si los datos son inválidos."""
        if not weight_kg or not height_cm or weight_kg <= 0 or height_cm <= 0:
            return None
        height_m = height_cm / 100 if height_cm > 10 else height_cm
        if height_m <= 0:
            return None
        return round(weight_kg / (height_m ** 2), 1)

    @staticmethod
    def classify_bmi_adult(bmi: float) -> str:
        """Clasificación OMS para adultos (>=18 años)."""
        if bmi < 18.5:
            return "bajo peso"
        elif bmi < 25.0:
            return "peso normal"
        elif bmi < 30.0:
            return "sobrepeso"
        elif bmi < 35.0:
            return "obesidad grado I"
        elif bmi < 40.0:
            return "obesidad grado II"
        else:
            return "obesidad grado III"

    @staticmethod
    def build_referential_message(snapshot: ProfileSnapshot) -> Optional[str]:
        """Construye un mensaje referencial de IMC a partir de un ProfileSnapshot."""
        weight = snapshot.measurements.weight_kg
        height = snapshot.measurements.height_cm
        age = snapshot.measurements.age_years

        if not weight or not height:
            return None

        bmi = NutritionAssessmentService.compute_bmi(weight, height)
        if bmi is None:
            return None

        height_m = height / 100 if height > 10 else height

        if age is not None and age < 18:
            return (
                f"📊 *IMC referencial*: ~{bmi} (Peso: {weight}kg, Talla: {height_m:.2f}m)\n"
                f"En menores de 18 años, la interpretación del IMC depende de la edad y el sexo, "
                f"por lo que te recomendamos consultar con tu profesional de salud para una evaluación adecuada."
            )

        category = NutritionAssessmentService.classify_bmi_adult(bmi)
        return (
            f"📊 *IMC referencial*: ~{bmi} (Peso: {weight}kg, Talla: {height_m:.2f}m)\n"
            f"Esto corresponde a la categoría de *{category}* según la OMS.\n"
            f"_Recuerda que esto es orientación referencial y no reemplaza una evaluación profesional._ 🏥"
        )

    @staticmethod
    def build_referential_message_from_flat(profile_flat: dict) -> Optional[str]:
        """Construye mensaje referencial directamente desde un dict de perfil plano."""
        weight = profile_flat.get("peso_kg")
        height = profile_flat.get("altura_cm")
        age = profile_flat.get("edad")

        if not weight or not height:
            return None

        try:
            weight = float(weight)
            height = float(height)
            age = int(age) if age else None
        except (ValueError, TypeError):
            return None

        bmi = NutritionAssessmentService.compute_bmi(weight, height)
        if bmi is None:
            return None

        height_m = height / 100 if height > 10 else height

        if age is not None and age < 18:
            return (
                f"📊 *IMC referencial*: ~{bmi} (Peso: {weight}kg, Talla: {height_m:.2f}m)\n"
                f"En menores de 18 años, la interpretación del IMC depende de la edad y el sexo, "
                f"por lo que te recomendamos consultar con tu profesional de salud para una evaluación adecuada."
            )

        category = NutritionAssessmentService.classify_bmi_adult(bmi)
        return (
            f"📊 *IMC referencial*: ~{bmi} (Peso: {weight}kg, Talla: {height_m:.2f}m)\n"
            f"Esto corresponde a la categoría de *{category}* según la OMS.\n"
            f"_Recuerda que esto es orientación referencial y no reemplaza una evaluación profesional._ 🏥"
        )
