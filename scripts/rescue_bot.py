import asyncio
from sqlalchemy import text
from infrastructure.db.connection import get_session_factory

async def rescue():
    print("🚀 Iniciando rescate de NutriBot...")
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            # Resetear incoming bloqueados.
            incoming = await session.execute(
                text(
                    """
                    UPDATE incoming_messages
                    SET status='pending',
                        locked_at=NULL,
                        updated_at=TIMEZONE('America/Lima', NOW())
                    WHERE status='processing'
                    """
                )
            )
            print(f"✅ Incoming liberados: {incoming.rowcount}")

            # Resetear outbox en envío/procesamiento.
            outbox = await session.execute(
                text(
                    """
                    UPDATE outgoing_messages
                    SET status='pending',
                        locked_at=NULL,
                        updated_at=TIMEZONE('America/Lima', NOW())
                    WHERE status IN ('processing', 'sending')
                    """
                )
            )
            print(f"✅ Outbox liberados: {outbox.rowcount}")
            
    print("🏁 Rescate completado. NutriBot está listo para procesar de nuevo.")

if __name__ == "__main__":
    asyncio.run(rescue())
