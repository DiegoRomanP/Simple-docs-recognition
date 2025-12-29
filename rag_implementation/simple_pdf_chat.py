"""
Versión simplificada para empezar rápido con PDFs
"""
from pdf_rag_system import PDFRAGSystem

class SimplePDFChat:
    @staticmethod
    def quick_start(pdf_path: str, model: str = "mistral"):
        # Verificar Ollama
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model not in result.stdout:
                print(f"⚠️  Modelo {model} no encontrado en Ollama. Ejecuta: ollama pull {model}")
                return
        except FileNotFoundError:
            print("⚠️  Ollama no encontrado. Instala desde: https://ollama.com/")
            return

        rag = PDFRAGSystem(model_name=model)
        rag.load_and_process_pdf(pdf_path)
        rag.chat_session()