"""
Archivo de prueba para las implementaciones RAG
"""
import os
from ultra_fast_markdown_rag import UltraFastMarkdownRAG
from pdf_rag_system import PDFRAGSystem
from simple_pdf_chat import SimplePDFChat

def test_markdown_rag():
    print("\n" + "="*60)
    print("🔷 PRUEBA: RAG para Markdown")
    print("="*60)
    
    # IMPORTANTE: Cambia esto por una ruta real en tu PC
    markdown_file = "README.md"  # Usamos el README como ejemplo
    
    if not os.path.exists(markdown_file):
        print(f"⚠️  Crea un archivo '{markdown_file}' para probar esta sección.")
        return

    md_rag = UltraFastMarkdownRAG(model_size="small")
    md_rag.index_markdown(markdown_file)
    
    preguntas = ["¿De qué trata este proyecto?", "Resumen"]
    for p in preguntas:
        print(f"\n📝 Pregunta: {p}")
        res = md_rag.query(p, k=2)
        for r in res:
            print(f"  > {r['header']}: {r['content'][:100]}...")

def test_pdf_rag():
    print("\n" + "="*60)
    print("🔶 PRUEBA: RAG para PDF")
    print("="*60)
    
    # IMPORTANTE: Cambia esto por una ruta real
    pdf_file = "docs/tu_documento.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"⚠️  Coloca un PDF en '{pdf_file}' para probar esta sección.")
        return
        
    print("\n📌 Iniciando chat...")
    SimplePDFChat.quick_start(pdf_file, model="mistral")

if __name__ == "__main__":
    test_markdown_rag()
    # test_pdf_rag()