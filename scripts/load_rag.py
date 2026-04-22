import asyncio
import os
import sys
import gc
from typing import List
from pypdf import PdfReader
from openai import AsyncOpenAI
from sqlalchemy import text
from pathlib import Path

# Add the parent directory to Python path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from infrastructure.db.connection import get_session_factory
from config import get_settings

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

def chunk_text(text_body: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text_body.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

async def process_page(session, client, doc_id, page_num, page_text, settings, start_fragment_num):
    chunks = chunk_text(page_text, chunk_size=350, overlap=50)
    print(f"➡️ Procesando página {page_num} ({len(chunks)} fragmentos)...")
    
    current_frag_num = start_fragment_num
    added_count = 0
    for j, chunk in enumerate(chunks):
        if len(chunk) < 40: continue
        
        emb_resp = await client.embeddings.create(
            input=chunk,
            model=settings.openai_embedding_model
        )
        vector = emb_resp.data[0].embedding
        vector_str = "[" + ",".join(map(str, vector)) + "]"
        
        await session.execute(text(
            """
            INSERT INTO fragmentos_rag (documento_id, contenido, embedding, pagina, numero_fragmento, metadata)
            VALUES (:did, :content, CAST(:v AS vector), :pg, :num, '{}'::jsonb)
            """
        ), {
            "did": doc_id,
            "content": chunk,
            "v": vector_str,
            "pg": page_num,
            "num": current_frag_num + 1
        })
        current_frag_num += 1
        added_count += 1
    return added_count, current_frag_num

async def main():
    print("🚀 Iniciando proceso de carga RAG (V4 - Numeración Global + Resumible)...")
    try:
        settings = get_settings()
        pdf_path = os.getenv("PDF_FILE", "docs/guias_alimentarias.pdf")
        if not os.path.isabs(pdf_path):
            pdf_path = str(Path(__file__).resolve().parent.parent / pdf_path)

        if not os.path.exists(pdf_path):
            print(f"❌ Error: El archivo PDF no existe en {pdf_path}")
            return

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"📄 PDF: {pdf_path} | Total páginas: {total_pages}")
        
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        factory = get_session_factory()
        
        doc_name = "Guías Alimentarias Perú (CENAN)"
        doc_id = None
        
        # 1. Obtener doc_id
        async with factory() as session:
            res = await session.execute(text("SELECT id FROM documentos_rag WHERE nombre = :n"), {"n": doc_name})
            doc_id = res.scalar()
            if not doc_id:
                res = await session.execute(text("INSERT INTO documentos_rag (nombre, tipo_fuente, estado) VALUES (:n, 'pdf', 'activo') RETURNING id"), {"n": doc_name})
                doc_id = res.scalar()
            await session.commit()

        # 2. Bucle de páginas con sesiones frescas
        total_added = 0
        for i in range(total_pages):
            page_num = i + 1
            
            # Verificar si ya existe en una sesión rápida
            async with factory() as session:
                res = await session.execute(text("SELECT 1 FROM fragmentos_rag WHERE documento_id = :did AND pagina = :p LIMIT 1"), {"did": doc_id, "p": page_num})
                if res.scalar():
                    if page_num % 10 == 0: print(f"⏩ Salto pág {page_num} (ya existe)")
                    continue

                # Si no existe, obtener el max numero_fragmento actual para continuar la serie
                res_max = await session.execute(text("SELECT COALESCE(MAX(numero_fragmento), 0) FROM fragmentos_rag WHERE documento_id = :did"), {"did": doc_id})
                max_frag = res_max.scalar()

            # Procesar página
            try:
                page_text = reader.pages[i].extract_text()
                if not page_text or not page_text.strip():
                    print(f"⚠️ Pág {page_num}: Sin texto.")
                    continue
                    
                async with factory() as session:
                    added, new_max = await process_page(session, client, doc_id, page_num, page_text, settings, max_frag)
                    await session.commit()
                    total_added += added
                
                gc.collect()
                await asyncio.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error en página {page_num}: {str(e)}")
                # Si falló a mitad de página, la siguiente iteración volverá a intentar esta página porque el commit no ocurrió (o falló)
                continue
            
        print(f"✅ Proceso finalizado. Total fragmentos añadidos en esta sesión: {total_added}")

    except Exception as e:
        print(f"🔥 Error fatal: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
