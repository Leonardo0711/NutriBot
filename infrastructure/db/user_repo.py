"""
Nutribot Backend — User Repository (SQLAlchemy)
Implementa el puerto UserRepository con acceso directo a SQL.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import User
from domain.ports import UserRepository
from .connection import get_session_factory


class SqlAlchemyUserRepository(UserRepository):

    async def get_or_create(self, phone: str) -> User:
        """
        Obtiene o crea un usuario por número de WhatsApp.
        Garantiza que siempre exista su conversation_state asociado (incluso legacy).
        Usa una sesión corta propia — no retiene transacciones abiertas.
        """
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                # 1. Upsert del usuario
                await session.execute(
                    text("""
                        INSERT INTO usuarios (whatsapp_jid, numero_whatsapp)
                        VALUES (:jid, :phone)
                        ON CONFLICT (numero_whatsapp) DO NOTHING
                    """),
                    {"jid": f"{phone}@s.whatsapp.net", "phone": phone},
                )

                # 2. Leer el usuario (nuevo o existente)
                result = await session.execute(
                    text("SELECT id, numero_whatsapp FROM usuarios WHERE numero_whatsapp = :phone"),
                    {"phone": phone},
                )
                row = result.fetchone()

                # 3. Garantizar conversation_state para este usuario
                await session.execute(
                    text("""
                        INSERT INTO conversation_state (usuario_id)
                        VALUES (:uid)
                        ON CONFLICT (usuario_id) DO NOTHING
                    """),
                    {"uid": row.id},
                )

                # 4. Garantizar registros en las tablas de "memoria" para nuevos usuarios
                # Esto evita errores de "record not found" en los servicios.
                await session.execute(
                    text("""
                        INSERT INTO memoria_chat (usuario_id)
                        VALUES (:uid)
                        ON CONFLICT (usuario_id) DO NOTHING
                    """),
                    {"uid": row.id},
                )
                await session.execute(
                    text("""
                        INSERT INTO perfil_nutricional (usuario_id)
                        VALUES (:uid)
                        ON CONFLICT (usuario_id) DO NOTHING
                    """),
                    {"uid": row.id},
                )
                # NO insertar en formulario_en_progreso aquí:
                # requiere formulario_id NOT NULL y se crea desde SurveyService
                # cuando el flujo de encuesta realmente inicia.

            return User(id=row.id, numero_whatsapp=row.numero_whatsapp)
