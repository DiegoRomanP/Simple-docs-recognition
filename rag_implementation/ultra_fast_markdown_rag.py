"""
Sistema RAG HIPER-OPTIMIZADO para Markdown
"""

import re
from typing import List, Dict, Optional
import time

# Dependencias ligeras
import markdown
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

class UltraFastMarkdownRAG:
    """
    RAG para Markdown: 10-100x más rápido que PDF
    """
    
    def __init__(self, model_size: str = "small", persist_directory: Optional[str] = None):
        """
        Inicializa con modelo pequeño pero efectivo.
        """
        self.embedding_models = {
            "tiny": "sentence-transformers/all-MiniLM-L6-v2",
            "small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "base": "intfloat/multilingual-e5-base"
        }
        
        print(f"🚀 Cargando modelo {model_size}...")
        start = time.time()
        
        # Embeddings
        self.embedder = SentenceTransformer(
            self.embedding_models.get(model_size, self.embedding_models["small"]),
            device='cpu'
        )
        
        # --- CORRECCIÓN PRINCIPAL AQUÍ ---
        # Inicialización moderna de ChromaDB (v0.4.x+)
        if persist_directory:
            print(f"Iniciando ChromaDB Persistente en: {persist_directory}")
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            print("Iniciando ChromaDB en Memoria (volátil)")
            self.chroma_client = chromadb.Client()
        
        # Obtener o crear colección
        self.collection = self.chroma_client.get_or_create_collection(name="markdown_docs")
        
        print(f"✅ Sistema listo en {time.time()-start:.2f}s")
    
    def parse_markdown_semantic(self, md_content: str) -> List[Dict]:
        """Parseo semántico de Markdown."""
        html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        soup = BeautifulSoup(html, 'html.parser')
        
        chunks = []
        current_chunk = ""
        current_header = ""
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'code', 'pre']):
            tag_name = element.name
            text = element.get_text().strip()
            
            if not text: continue
            
            if tag_name.startswith('h'):
                if current_chunk:
                    chunks.append({"content": current_chunk, "header": current_header, "type": "section"})
                current_header = text
                current_chunk = f"# {text}\n\n"
            
            elif tag_name in ['p', 'li']:
                if len(current_chunk) + len(text) < 1500:
                    current_chunk += text + "\n"
                else:
                    if current_chunk:
                        chunks.append({"content": current_chunk, "header": current_header, "type": "section"})
                    current_chunk = text + "\n"
            
            elif tag_name in ['code', 'pre']:
                if current_chunk:
                    chunks.append({"content": current_chunk, "header": current_header, "type": "section"})
                chunks.append({"content": f"```\n{text}\n```", "header": "Código", "type": "code"})
                current_chunk = ""
        
        if current_chunk:
            chunks.append({"content": current_chunk, "header": current_header, "type": "section"})
        
        return chunks
    
    def index_markdown(self, md_path: str) -> None:
        """Indexa un archivo Markdown."""
        print(f"📄 Indexando {md_path}...")
        start = time.time()
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"❌ Error: Archivo no encontrado: {md_path}")
            return

        chunks = self.parse_markdown_semantic(content)
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [{"header": c["header"], "type": c["type"], "source": md_path} for c in chunks]
        ids = [f"doc_{time.time()}_{i}" for i in range(len(texts))]
        
        embeddings = self.embedder.encode(texts, batch_size=32, normalize_embeddings=True)
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Indexado: {len(chunks)} chunks en {time.time() - start:.2f}s")
    
    def query(self, question: str, k: int = 5) -> List[Dict]:
        """Búsqueda semántica."""
        query_embedding = self.embedder.encode([question]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results['documents']:
            for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1):
                formatted.append({
                    "rank": i,
                    "content": doc,
                    "header": meta.get('header', ''),
                    "score": round(1 - dist, 3)
                })
        return formatted