"""
Archivo de prueba para las implementaciones RAG
Demuestra el uso de ambos sistemas: Markdown y PDF
"""

from ultra_fast_markdown_rag import UltraFastMarkdownRAG
from pdf_rag_system import PDFRAGSystem
from simple_pdf_chat import SimplePDFChat


def test_markdown_rag():
    """
    Prueba la implementación RAG para archivos Markdown
    """
    print("\n" + "="*60)
    print("🔷 PRUEBA: RAG para Markdown")
    print("="*60)
    
    # Inicializar el sistema RAG para Markdown
    md_rag = UltraFastMarkdownRAG(model_size="small")
    
    # RUTA DE EJEMPLO - Ajusta según tus archivos
    # Descomenta y ajusta la ruta a tu archivo markdown
    markdown_file = "markdowns/nota_prueba.md"
    
    # Ejemplo de uso (comentado porque necesitas un archivo real)
    
    # Indexar el archivo Markdown
    md_rag.index_markdown(markdown_file)
    
    # Hacer consultas
    preguntas = [
        "¿Cuál es el tema principal del documento?",
        "¿Qué información sobre código contiene?",
        "Resume las secciones principales"
    ]
    
    for pregunta in preguntas:
        print(f"\n📝 Pregunta: {pregunta}")
        resultados = md_rag.query(pregunta, k=3)
        
        print("\n🔍 Resultados:")
        for res in resultados:
            print(f"\n  Rank {res['rank']}: {res['header']}")
            print(f"  Score: {res['score']}")
            print(f"  Contenido: {res['content'][:200]}...")
    
    
    print("\n⚠️  Configura la ruta del archivo Markdown en la línea 21")
    print("    Luego descomenta las líneas 25-41 para ejecutar la prueba")


def test_pdf_rag():
    """
    Prueba la implementación RAG para archivos PDF
    """
    print("\n" + "="*60)
    print("🔶 PRUEBA: RAG para PDF")
    print("="*60)
    
    # RUTA DE EJEMPLO - Ajusta según tus archivos
    # Descomenta y ajusta la ruta a tu archivo PDF
    pdf_file = "pdfs/lecturas-para-todos-los-dias.pdf"
    
    # Ejemplo de uso (comentado porque necesitas un archivo real y Ollama configurado)
    # Opción 1: Uso rápido con SimplePDFChat
    print("\n📌 Opción 1: Inicio rápido con chat interactivo")
    SimplePDFChat.quick_start(pdf_file, model="mistral")
    
    # Opción 2: Uso avanzado con PDFRAGSystem
    print("\n📌 Opción 2: Uso programático")
    pdf_rag = PDFRAGSystem(
        model_name="llama3:8b",
        embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Cargar y procesar el PDF
    pdf_rag.load_and_process_pdf(pdf_file)
    
    # Hacer preguntas programáticamente
    preguntas = [
        "¿Cuáles son los puntos principales del documento?",
        "¿Qué información relevante contiene?",
        "Resume el contenido del PDF"
    ]
    
    for pregunta in preguntas:
        print(f"\n📝 Pregunta: {pregunta}")
        resultado = pdf_rag.ask_question(pregunta)
        
        print("\n🤖 Respuesta:")
        print(resultado["answer"])
        
        print("\n📚 Fuentes:")
        for i, source in enumerate(resultado["sources"], 1):
            print(f"  {i}. Página {source['page']}: {source['content']}")
    
    print("\n⚠️  Configura la ruta del archivo PDF en la línea 54")
    print("    Asegúrate de tener Ollama instalado y un modelo descargado")
    print("    Luego descomenta las líneas 58-93 para ejecutar la prueba")


def main():
    """
    Función principal para ejecutar las pruebas
    """
    print("\n" + "🚀"*30)
    print("  SISTEMA RAG - PRUEBAS DE IMPLEMENTACIÓN")
    print("🚀"*30)
    
    print("\n📋 Este archivo demuestra el uso de:")
    print("  1. UltraFastMarkdownRAG - Sistema RAG para archivos Markdown")
    print("  2. PDFRAGSystem - Sistema RAG completo para PDFs")
    print("  3. SimplePDFChat - Interfaz simplificada para PDFs")
    
    # Ejecutar pruebas
    test_markdown_rag()
    test_pdf_rag()
    
    print("\n" + "="*60)
    print("✅ Configuración de pruebas completa")
    print("="*60)
    print("\n💡 Próximos pasos:")
    print("  1. Ajusta las rutas de archivos en las funciones de prueba")
    print("  2. Para PDF: Instala Ollama (https://ollama.com/)")
    print("  3. Para PDF: Descarga un modelo (ej: ollama pull mistral)")
    print("  4. Descomenta el código de ejemplo y ejecuta")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
