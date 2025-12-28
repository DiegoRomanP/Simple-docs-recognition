"""
Versión simplificada para empezar rápido con PDFs
"""

from pdf_rag_system import PDFRAGSystem


class SimplePDFChat:
    """Versión simplificada para empezar rápido"""
    
    @staticmethod
    def quick_start(pdf_path: str, model: str = "mistral"):
        """
        Inicio rápido: procesa PDF y abre chat.
        
        Args:
            pdf_path: Ruta al PDF
            model: Modelo a usar (llama3, mistral, etc.)
        """
        # Verificar que Ollama está instalado y el modelo descargado
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model not in result.stdout:
                print(f"⚠️  Modelo {model} no encontrado en Ollama.")
                print(f"Ejecuta: ollama pull {model}")
                return
        except:
            print("⚠️  Ollama no encontrado. Instala desde: https://ollama.com/")
            return
        
        # Crear y usar el sistema
        rag = PDFRAGSystem(model_name=model)
        rag.load_and_process_pdf(pdf_path)
        rag.chat_session()
