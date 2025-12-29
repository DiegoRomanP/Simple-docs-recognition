import os
import sys
from PIL import Image
import re

# ==============================================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# ==============================================================================
# Aseguramos que el directorio raíz esté en el path para importar los submódulos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# --- IMPORTACIONES DE TUS MÓDULOS ---
try:
    # Módulo 1: Layout y Preprocesamiento
    from preproccesing.layout_config import LayoutConfig
    from preproccesing.document_layout_pipeline import DocumentLayoutPipeline
    
    # Módulo 2: OCR y Transcripción
    from ocr.ContentExtractor import ContentExtractor
    
    # Módulo 3: RAG (Retrieval-Augmented Generation)
    from rag_implementation.ultra_fast_markdown_rag import UltraFastMarkdownRAG
except ImportError as e:
    print("\n❌ ERROR CRÍTICO DE IMPORTACIÓN")
    print(f"No se pudo importar un submódulo: {e}")
    print("Asegúrate de que existan archivos '__init__.py' en las carpetas 'preproccesing', 'ocr' y 'rag_implementation'.")
    sys.exit(1)

# --- RUTAS GLOBALES ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MARKDOWNS_DIR = os.path.join(BASE_DIR, "markdowns")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_permanent_db") # Base de datos persistente para el RAG

# Asegurar directorios de salida
os.makedirs(MARKDOWNS_DIR, exist_ok=True)

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def extract_sort_key(filepath):
    """
    Ayuda a ordenar los recortes numéricamente.
    Extrae el número 'N' de archivos tipo '..._crop_N_Label.jpg'
    """
    filename = os.path.basename(filepath)
    match = re.search(r'_crop_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 9999 # Si no encuentra número, va al final

def extract_label_from_filename(filepath):
    """
    Extrae la etiqueta (Label) del nombre del archivo.
    Ej: 'imagen1_crop_2_Title.jpg' -> 'Title'
    """
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    # Asume formato: base_crop_index_Label
    try:
        parts = name_no_ext.split('_')
        crop_idx = parts.index('crop')
        # El label es todo lo que sigue después del índice numérico
        label = "_".join(parts[crop_idx + 2:])
        return label
    except ValueError:
        return "Text" # Fallback genérico

# ==============================================================================
# CLASE PRINCIPAL ORQUESTADORA
# ==============================================================================
class FullStackPipeline:
    def __init__(self, use_gpu=False):
        print("\n" + "="*60)
        print("🚀 INICIALIZANDO MOTORES IA (Esto puede tardar un poco)...")
        print("="*60)
        self.use_gpu = use_gpu

        # 1. INICIALIZAR MOTOR DE LAYOUT (Detectron2/LayoutParser)
        print("\n[1/3] 📐 Cargando modelo de Layout...")
        # AJUSTA ESTO SI USAS OTRO MODELO (ej. NewspaperNavigator)
        model_subfolder = "PubLayNet_faster" 
        config_path = os.path.join(MODELS_DIR, model_subfolder, "config.yml")
        model_path = os.path.join(MODELS_DIR, model_subfolder, "model_final.pth")
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
             raise FileNotFoundError(f"No se encontraron los archivos del modelo en: {os.path.join(MODELS_DIR, model_subfolder)}")

        layout_config = LayoutConfig(
            config_path=config_path,
            model_path=model_path,
            label_map_name="PubLayNet", # Ajusta según el modelo que uses
            score_threshold=0.75,
        )
        self.layout_pipeline = DocumentLayoutPipeline(layout_config)

        # 2. INICIALIZAR MOTOR OCR (TrOCR + LaTeX-OCR)
        print("\n[2/3] 👁️  Cargando modelos de OCR/HTR...")
        self.ocr_engine = ContentExtractor(use_gpu=self.use_gpu)

        # 3. INICIALIZAR MOTOR RAG (Embeddings + Vector DB)
        print("\n[3/3] 🧠 Inicializando sistema RAG (Memoria)...")
        # Usamos persistencia para no tener que re-indexar si reiniciamos el script
        self.rag_engine = UltraFastMarkdownRAG(
            model_size="small", 
            persist_directory=CHROMA_DB_DIR
        )
        
        print("\n✅ ¡TODOS LOS SISTEMAS LISTOS!")

    def run(self, image_filename):
        """Ejecuta el flujo completo para una imagen."""
        image_path = os.path.join(IMAGES_DIR, image_filename)
        if not os.path.exists(image_path):
            print(f"❌ Error: La imagen '{image_filename}' no existe en la carpeta 'images/'.")
            return

        print(f"\n" + "="*60)
        print(f"📂 PROCESANDO: {image_filename}")
        print("="*60)

        base_name = os.path.splitext(image_filename)[0]
        md_output_path = os.path.join(MARKDOWNS_DIR, f"{base_name}.md")

        # ==========================================
        # ETAPA A: Layout y Recorte
        # ==========================================
        print("\n--- ETAPA A: Análisis Visual y Segmentación ---")
        # El pipeline ya se encarga de guardar los recortes en images_processed/
        layout_results = self.layout_pipeline.process_document(
            image_path=image_path,
            preprocess=True # Recomendado True para pizarras
        )
        
        crop_paths = layout_results.get("crops", [])
        if not crop_paths:
            print("❌ Error: No se generaron recortes. Revisa la detección de layout.")
            return

        print(f"✅ Se generaron {len(crop_paths)} recortes.")
        
        # Ordenar los recortes para que el texto tenga sentido (1, 2, 3...)
        crop_paths.sort(key=extract_sort_key)

        # ==========================================
        # ETAPA B: Transcripción (OCR -> Markdown)
        # ==========================================
        print("\n--- ETAPA B: Transcripción Inteligente (OCR/HTR) ---")
        markdown_content = f"# Transcripción de: {image_filename}\n\n"
        
        print("Procesando fragmentos...")
        for i, crop_path in enumerate(crop_paths, 1):
            label = extract_label_from_filename(crop_path)
            print(f"  -> [{i}/{len(crop_paths)}] Tipo: '{label}' ... ", end="", flush=True)
            
            try:
                img = Image.open(crop_path).convert("RGB")
                # Usamos el método que agregamos recientemente a ContentExtractor
                texto = self.ocr_engine.procesar_imagen_ya_recortada(img, label)
                markdown_content += texto + "\n"
                print("OK")
            except Exception as e:
                print(f"ERROR ({e})")
                markdown_content += f"\n[Error al leer fragmento {i}]\n"

        # Guardar Markdown
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"\n✅ Markdown generado en: {md_output_path}")

        # ==========================================
        # ETAPA C: Asimilación de Conocimiento (RAG)
        # ==========================================
        print("\n--- ETAPA C: Indexación en Base de Conocimiento (RAG) ---")
        self.rag_engine.index_markdown(md_output_path)
        print("✅ Documento indexado y listo para consultas.")

        # ==========================================
        # ETAPA D: Interacción
        # ==========================================
        self.iniciar_chat_interactivo()

    def iniciar_chat_interactivo(self):
        print("\n" + "="*60)
        print("🤖 CHAT CON TUS DOCUMENTOS")
        print("   (Escribe 'salir' para terminar)")
        print("="*60)
        
        # Verifica que tengas Ollama corriendo si usas answer_question
        llm_model = "mistral" # O "llama3", el que tengas en Ollama

        while True:
            try:
                pregunta = input("\n📝 Tú: ").strip()
                if not pregunta: continue
                if pregunta.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                # Llamada al RAG para obtener respuesta generada
                print("Thinking...")
                respuesta = self.rag_engine.answer_question(pregunta, model_name=llm_model)
                
                print(f"\n🤖 IA:\n{respuesta}")
                
            except KeyboardInterrupt:
                print("\nSalida forzada.")
                break
            except Exception as e:
                print(f"❌ Error en el chat: {e}")
                print("Consejo: Asegúrate de que Ollama esté corriendo (`ollama serve`) si usas generación.")

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    # 1. Configuración inicial (CPU vs GPU)
    # Pon True si tienes CUDA configurado correctamente en PyTorch
    USE_GPU = False 

    # 2. Nombre de la imagen a procesar (debe estar en la carpeta 'images/')
    IMAGEN_A_PROCESAR = "imagen5.jpeg" 

    try:
        # Inicializar el stack completo
        full_stack = FullStackPipeline(use_gpu=USE_GPU)
        
        # Ejecutar el flujo
        full_stack.run(IMAGEN_A_PROCESAR)
        
    except Exception as e:
        print(f"\n❌ Error fatal en el pipeline principal:\n{e}")
