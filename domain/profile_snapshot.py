"""
Nutribot Backend - Profile Snapshot Domain Model
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


def _norm_items(values: Optional[Iterable[str]]) -> tuple[str, ...]:
    if not values:
        return tuple()
    clean = []
    seen = set()
    for value in values:
        if value is None:
            continue
        token = str(value).strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(token)
    return tuple(clean)


@dataclass(frozen=True)
class ProfileMeasurements:
    age_years: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

    @property
    def height_m(self) -> Optional[float]:
        if self.height_cm is None:
            return None
        if self.height_cm > 10:
            return round(self.height_cm / 100, 2)
        return round(self.height_cm, 2)


@dataclass(frozen=True)
class ProfileHealth:
    diet_type: Optional[str] = None
    allergies: tuple[str, ...] = field(default_factory=tuple)
    diseases: tuple[str, ...] = field(default_factory=tuple)
    food_restrictions: tuple[str, ...] = field(default_factory=tuple)
    nutrition_goal: Optional[str] = None

    @property
    def allergies_text(self) -> Optional[str]:
        return ", ".join(self.allergies) if self.allergies else None

    @property
    def diseases_text(self) -> Optional[str]:
        return ", ".join(self.diseases) if self.diseases else None

    @property
    def restrictions_text(self) -> Optional[str]:
        return ", ".join(self.food_restrictions) if self.food_restrictions else None


@dataclass(frozen=True)
class ProfileLocation:
    region: Optional[str] = None
    province: Optional[str] = None
    district: Optional[str] = None

    @property
    def best_location_text(self) -> Optional[str]:
        return self.district or self.province or self.region


@dataclass(frozen=True)
class ProfileSnapshot:
    user_id: int
    measurements: ProfileMeasurements
    health: ProfileHealth
    location: ProfileLocation
    skipped_fields: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def from_row(row: dict) -> "ProfileSnapshot":
        raw_skips = row.get("skipped_fields") or {}
        skip_fields = frozenset(k for k, v in raw_skips.items() if bool(v))
        return ProfileSnapshot(
            user_id=int(row.get("usuario_id") or 0),
            measurements=ProfileMeasurements(
                age_years=row.get("edad"),
                weight_kg=row.get("peso_kg"),
                height_cm=row.get("altura_cm"),
            ),
            health=ProfileHealth(
                diet_type=row.get("tipo_dieta"),
                allergies=_norm_items(row.get("alergias_items")),
                diseases=_norm_items(row.get("enfermedades_items")),
                food_restrictions=_norm_items(row.get("restricciones_items")),
                nutrition_goal=row.get("objetivo_nutricional"),
            ),
            location=ProfileLocation(
                region=row.get("region"),
                province=row.get("provincia"),
                district=row.get("distrito"),
            ),
            skipped_fields=skip_fields,
        )

    def value_for_step(self, step_code: str):
        mapping = {
            "edad": self.measurements.age_years,
            "peso_kg": self.measurements.weight_kg,
            "altura_cm": self.measurements.height_cm,
            "alergias": self.health.allergies_text,
            "enfermedades": self.health.diseases_text,
            "restricciones_alimentarias": self.health.restrictions_text,
            "tipo_dieta": self.health.diet_type,
            "objetivo_nutricional": self.health.nutrition_goal,
            "region": self.location.region,
            "provincia": self.location.province,
            "distrito": self.location.district,
        }
        return mapping.get(step_code)

    def to_legacy_dict(self) -> dict:
        return {
            "usuario_id": self.user_id,
            "edad": self.measurements.age_years,
            "peso_kg": self.measurements.weight_kg,
            "altura_cm": self.measurements.height_cm,
            "tipo_dieta": self.health.diet_type,
            "alergias": self.health.allergies_text,
            "enfermedades": self.health.diseases_text,
            "restricciones_alimentarias": self.health.restrictions_text,
            "objetivo_nutricional": self.health.nutrition_goal,
            "region": self.location.region,
            "provincia": self.location.province,
            "distrito": self.location.district,
            "skipped_fields": {field: True for field in self.skipped_fields},
        }
