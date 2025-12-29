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
    
    markdown_file = "README.md"
    
    if not os.path.exists(markdown_file):
        print(f"⚠️  Crea '{markdown_file}' para probar.")
        return

    md_rag = UltraFastMarkdownRAG(model_size="small")
    md_rag.index_markdown(markdown_file)
    
    preguntas = ["¿De qué trata este proyecto?", "Resumen de la estructura"]
    
    for p in preguntas:
        print(f"\n📝 Pregunta: {p}")
        
        # AHORA LLAMAMOS A LA FUNCIÓN QUE GENERA LA RESPUESTA
        respuesta = md_rag.answer_question(p, model_name="mistral") # O llama3
        
        print("\n🤖 Respuesta:")
        print(respuesta)

def test_pdf_rag():
    print("\n" + "="*60)
    print("🔶 PRUEBA: RAG para PDF")
    print("="*60)
    
    # IMPORTANTE: Cambia esto por una ruta real
    pdf_file = "docs/prueba.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"⚠️  Coloca un PDF en '{pdf_file}' para probar esta sección.")
        return
        
    print("\n📌 Iniciando chat...")
    SimplePDFChat.quick_start(pdf_file, model="mistral")

if __name__ == "__main__":
    test_markdown_rag()
    test_pdf_rag()