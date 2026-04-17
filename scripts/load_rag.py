import asyncio
import os
from typing import List
from pypdf import PdfReader
from openai import AsyncOpenAI
from sqlalchemy import text

# Add the parent directory to Python path to allow imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from infrastructure.db.connection import get_session_factory
from config import get_settings

def chunk_text(text_body: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    words = text_body.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

async def main():
    settings = get_settings()
    pdf_path = os.getenv("PDF_FILE", "docs/guias_alimentarias.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} no encontrado.")
        return

    print(f"📄 Procesando PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    factory = get_session_factory()
    
    async with factory() as session:
        # 1. Verificar/Ajustar columnas si es necesario (sin crear tablas nuevas)
        # El dbdiagram.txt dice 'nombre', 'tipo_fuente' en documentos_rag
        # y 'metadata', 'numero_fragmento' en fragmentos_rag.
        
        # Check si ya existe el doc para no duplicar
        check_doc = await session.execute(
            text("SELECT id FROM documentos_rag WHERE nombre = :title"), 
            {"title": "Guías Alimentarias Perú (CENAN)"}
        )
        doc_id = check_doc.scalar()
        
        if not doc_id:
            print("📝 Registrando documento en documentos_rag...")
            res = await session.execute(text(
                "INSERT INTO documentos_rag (nombre, tipo_fuente, estado) VALUES (:title, 'pdf', 'activo') RETURNING id"
            ), {"title": "Guías Alimentarias Perú (CENAN)"})
            doc_id = res.scalar()
        else:
            print("📝 El documento ya existe, limpiando fragmentos antiguos...")
            await session.execute(text("DELETE FROM fragmentos_rag WHERE documento_id = :did"), {"did": doc_id})

        total_chunks = 0
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue
                
            chunks = chunk_text(page_text, chunk_size=300, overlap=50)
            print(f"  -> Página {i+1}: Generando {len(chunks)} fragmentos...")
            
            for j, chunk in enumerate(chunks):
                if len(chunk) < 50: continue
                
                # Get embeddings (OpenAI)
                try:
                    emb_resp = await client.embeddings.create(
                        input=chunk,
                        model=settings.openai_embedding_model
                    )
                    vector = emb_resp.data[0].embedding
                except Exception as e:
                    print(f"      ❌ Error OpenAI en chunk {j}: {e}")
                    continue

                # El embedding se guarda como string [v1, v2...] para que el CAST(:v AS vector) funcione en Postgres
                vector_str = "[" + ",".join(map(str, vector)) + "]"
                
                await session.execute(text(
                    """
                    INSERT INTO fragmentos_rag (documento_id, contenido, embedding, pagina, numero_fragmento, metadata)
                    VALUES (:did, :content, :v, :pg, :num, '{}'::jsonb)
                    """
                ), {
                    "did": doc_id,
                    "content": chunk,
                    "v": vector_str,
                    "pg": i + 1,
                    "num": total_chunks + 1
                })
                total_chunks += 1
        
        await session.commit()
    print(f"✅ RAG cargado exitosamente. Total fragmentos: {total_chunks}")

if __name__ == "__main__":
    asyncio.run(main())
