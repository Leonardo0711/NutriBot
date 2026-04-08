import asyncio
from sqlalchemy import text
from infrastructure.db.connection import get_session_factory

async def rescue():
    print("🚀 Iniciando rescate de NutriBot...")
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            # Resetear mensajes bloqueados
            result = await session.execute(
                text("UPDATE incoming_messages SET status='pending' WHERE status='processing'")
            )
            print(f"✅ Mensajes liberados: {result.rowcount}")
            
            # Limpiar jobs de extracción duplicados si los hay
            await session.execute(text("DELETE FROM extraction_jobs WHERE status='pending'"))
            print("✅ Cola de extracción limpiada.")
            
    print("🏁 Rescate completado. NutriBot está listo para procesar de nuevo.")

if __name__ == "__main__":
    asyncio.run(rescue())
