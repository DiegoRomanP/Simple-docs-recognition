"""
Sistema RAG HIPER-OPTIMIZADO para Markdown
"""

import re
from pathlib import Path
from typing import List, Dict
import time

# Dependencias MUCHO más ligeras que para PDF
import markdown
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class UltraFastMarkdownRAG:
    """
    RAG para Markdown: 10-100x más rápido que PDF
    """
    
    def __init__(self, model_size: str = "small"):
        """
        Inicializa con modelo pequeño pero efectivo.
        
        Args:
            model_size: "tiny", "small", "base"
        """
        self.embedding_models = {
            "tiny": "sentence-transformers/all-MiniLM-L6-v2",  # 80MB
            "small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 420MB
            "base": "intfloat/multilingual-e5-base"  # 1.1GB
        }
        
        print(f"🚀 Cargando modelo {model_size}...")
        start = time.time()
        
        # Embeddings MUCHO más rápidos (modelos pequeños)
        self.embedder = SentenceTransformer(
            self.embedding_models[model_size],
            device='cpu'  # Puede correr hasta en CPU
        )
        
        # ChromaDB en memoria (ultra rápido)
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=None,  # En memoria
            anonymized_telemetry=False
        ))
        
        self.collection = self.chroma_client.create_collection("markdown_docs")
        
        print(f"✅ Sistema listo en {time.time()-start:.2f}s")
    
    def parse_markdown_semantic(self, md_content: str) -> List[Dict]:
        """
        Parseo semántico inteligente de Markdown.
        Preserva estructura: headers, listas, código.
        """
        # Convertir a HTML para extracción estructurada
        html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        soup = BeautifulSoup(html, 'html.parser')
        
        chunks = []
        current_chunk = ""
        current_header = ""
        
        # Extraer elementos semánticos
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'code', 'pre']):
            tag_name = element.name
            text = element.get_text().strip()
            
            if not text:
                continue
            
            # Headers: crear nuevo chunk
            if tag_name.startswith('h'):
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "header": current_header,
                        "type": "section"
                    })
                
                current_header = text
                current_chunk = f"# {text}\n\n"
            
            # Párrafos y listas: añadir al chunk actual
            elif tag_name in ['p', 'li']:
                if len(current_chunk) + len(text) < 1500:  # Chunk size óptimo
                    current_chunk += text + "\n"
                else:
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk,
                            "header": current_header,
                            "type": "section"
                        })
                    current_chunk = text + "\n"
            
            # Código: chunk separado
            elif tag_name in ['code', 'pre']:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "header": current_header,
                        "type": "section"
                    })
                chunks.append({
                    "content": f"```\n{text}\n```",
                    "header": "Código",
                    "type": "code"
                })
                current_chunk = ""
        
        # Último chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "header": current_header,
                "type": "section"
            })
        
        return chunks
    
    def index_markdown(self, md_path: str) -> None:
        """
        Indexa un archivo Markdown con chunking semántico.
        """
        print(f"📄 Indexando {md_path}...")
        start = time.time()
        
        # Leer contenido
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parseo semántico
        chunks = self.parse_markdown_semantic(content)
        
        # Extraer textos para embeddings
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "header": chunk["header"],
                "type": chunk["type"],
                "source": md_path,
                "length": len(chunk["content"])
            }
            for chunk in chunks
        ]
        
        # Embeddings en BATCH (ultra rápido)
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # Añadir a ChromaDB
        ids = [f"doc_{i}_{hash(text)%10000}" for i, text in enumerate(texts)]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        elapsed = time.time() - start
        print(f"✅ Indexado: {len(chunks)} chunks en {elapsed:.2f}s")
        print(f"📊 Velocidad: {len(content)/elapsed/1000:.1f} KB/s")
    
    def query(self, question: str, k: int = 5) -> List[Dict]:
        """
        Búsqueda semántica ultra rápida.
        """
        # Embedding de la pregunta
        query_embedding = self.embedder.encode([question]).tolist()[0]
        
        # Buscar
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatear con contexto estructural
        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = 1 - dist  # Convertir a score
            
            # Añadir contexto del header
            header_info = f"Sección: {meta['header']}" if meta['header'] else ""
            
            formatted.append({
                "rank": i,
                "content": doc,
                "header": meta['header'],
                "type": meta['type'],
                "source": meta['source'],
                "score": round(similarity, 3),
                "context": f"{header_info}\n\n{doc[:500]}..."
            })
        
        return formatted
    
    def answer_with_context(self, question: str, context_chunks: List[str]) -> str:
        """
        Genera respuesta usando contextos de Markdown.
        Formatea el prompt para aprovechar la estructura MD.
        """
        context_text = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""Eres un asistente analizando documentos Markdown.

CONTEXTO DEL DOCUMENTO:
{context_text}

INSTRUCCIONES:
- El documento está en formato Markdown (# headers, **negrita**, etc.)
- Usa la estructura jerárquica para dar respuestas organizadas
- Si hay código, cítalo con ``` bloques de código
- Mantén el formato Markdown en tu respuesta cuando sea apropiado

PREGUNTA: {question}

RESPUESTA (en Markdown, estructurada):"""
        
        # Aquí integrarías tu LLM local (Ollama, etc.)
        # Por ahora retornamos un placeholder
        return f"**Respuesta basada en {len(context_chunks)} secciones relevantes:**\n\nConsulta procesada: '{question}'"