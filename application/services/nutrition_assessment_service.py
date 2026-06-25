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
    def build_initial_diet_guidance(snapshot: ProfileSnapshot) -> Optional[str]:
        """Orientación breve de dieta al completar el perfil básico."""
        if not snapshot:
            return None

        age = snapshot.measurements.age_years
        weight = snapshot.measurements.weight_kg
        height = snapshot.measurements.height_cm
        goal = (snapshot.health.nutrition_goal or "").lower()
        diseases_text = " ".join(snapshot.health.diseases).lower()
        restrictions = tuple(snapshot.health.allergies) + tuple(snapshot.health.food_restrictions)

        focus = (
            "una alimentación equilibrada tipo plato saludable: "
            "1/2 plato de verduras, 1/4 de proteína magra y 1/4 de carbohidrato saludable"
        )

        bmi = NutritionAssessmentService.compute_bmi(weight, height) if weight and height else None
        if age is not None and age < 18:
            focus = (
                "una alimentación variada y suficiente para tu etapa de crecimiento, "
                "con evaluación personalizada si deseas ajustar peso o composición corporal"
            )
        elif "bajar" in goal or "perder" in goal or (bmi is not None and bmi >= 25):
            focus = (
                "una dieta hipocalórica moderada y equilibrada, con porciones controladas, "
                "más verduras, proteína magra y carbohidratos integrales en cantidades medidas"
            )
        elif "ganar" in goal or "masa" in goal or (bmi is not None and bmi < 18.5):
            focus = (
                "una dieta con suficiente energía y proteína, repartida en comidas completas, "
                "para favorecer ganancia de masa de forma progresiva"
            )
        elif "diabetes" in diseases_text:
            focus = (
                "una dieta equilibrada con control de carbohidratos, evitando bebidas azucaradas "
                "y priorizando menestras, verduras y granos integrales"
            )

        notes: list[str] = []
        if "diabetes" in diseases_text and "carbohidratos" not in focus:
            notes.append("controlar carbohidratos y evitar bebidas azucaradas")
        if "hipertension" in diseases_text or "hipertensión" in diseases_text or "presion" in diseases_text or "presión" in diseases_text:
            notes.append("reducir sal y productos ultraprocesados")
        if "anemia" in diseases_text:
            notes.append("incluir alimentos ricos en hierro junto con vitamina C")
        if restrictions:
            notes.append("evitar tus alergias o restricciones registradas")

        guidance = f"🍽️ *Dieta sugerida*: por ahora te conviene seguir {focus}."
        if notes:
            guidance += "\nAdemás, considera: " + "; ".join(notes[:3]) + "."
        return guidance

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
