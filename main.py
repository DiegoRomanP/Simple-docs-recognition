import os
import sys
import re
from PIL import Image

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# ==============================================================================
# Obtener ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CORRECCIÓN CRÍTICA ---
# Agregamos no solo la raíz, sino también las subcarpetas al PATH del sistema.
# Esto permite que los scripts internos hagan "import layout_config" sin fallar.
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "preproccesing"))
sys.path.append(os.path.join(BASE_DIR, "extracting_text"))
sys.path.append(os.path.join(BASE_DIR, "rag_implementation"))

# --- DEFINICIÓN DE DIRECTORIOS CLAVE ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
IMAGES_PROCESSED_DIR = os.path.join(BASE_DIR, "images_processed")
MARKDOWNS_DIR = os.path.join(BASE_DIR, "markdowns")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db") 

# Asegurar que existan los directorios de salida
os.makedirs(MARKDOWNS_DIR, exist_ok=True)
os.makedirs(IMAGES_PROCESSED_DIR, exist_ok=True)

print("Cargando módulos del sistema...")

try:
    # AHORA SÍ funcionarán los imports internos
    from preproccesing.layout_config import LayoutConfig
    from preproccesing.document_layout_pipeline import DocumentLayoutPipeline
    
    from extracting_text.ContentExtractor import ContentExtractor
    from rag_implementation.ultra_fast_markdown_rag import UltraFastMarkdownRAG

except ImportError as e:
    print(f"\n❌ ERROR DE IMPORTACIÓN: {e}")
    print("Detalle: Python no encuentra un archivo interno. Con el sys.path.append agregado arriba debería funcionar.")
    sys.exit(1)

# ==============================================================================
# 2. FUNCIONES AUXILIARES
# ==============================================================================
def extract_sort_key(filepath):
    """
    Ordena los recortes numéricamente (ej: imagen1_crop_2_Title.jpg va antes que crop_10).
    """
    filename = os.path.basename(filepath)
    # Busca el patrón _crop_NUMERO_
    match = re.search(r'_crop_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return 9999

def extract_label_from_filename(filepath):
    """
    Recupera la etiqueta del nombre del archivo para saber qué modelo OCR usar.
    Ej: 'imagen1_crop_2_Figure.jpg' -> 'Figure'
    """
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    try:
        parts = name_no_ext.split('_')
        if 'crop' in parts:
            crop_idx = parts.index('crop')
            # El label es todo lo que sigue al número (ej: crop -> 2 -> Label)
            label = "_".join(parts[crop_idx + 2:])
            return label
    except Exception:
        pass
    return "Text" # Fallback por defecto

# ==============================================================================
# 3. CLASE PRINCIPAL: FullStackPipeline
# ==============================================================================
class FullStackPipeline:
    def __init__(self, use_gpu=False):
        print("\n" + "="*60)
        print("🚀 INICIALIZANDO PIPELINE DE DIGITALIZACIÓN")
        print("="*60)
        self.use_gpu = use_gpu

        # --- A. MOTOR DE LAYOUT ---
        print("\n[1/3] 📐 Configurando Motor de Layout...")
        
        # Ruta específica a tu modelo PubLayNet
        config_path = os.path.join(MODELS_DIR, "PubLayNet_faster", "config.yml")
        model_path = os.path.join(MODELS_DIR, "PubLayNet_faster", "model_final.pth")
        
        # Validación de archivos
        if not os.path.exists(config_path) or not os.path.exists(model_path):
             print(f"⚠️  Advertencia: No encontré el modelo en: {os.path.join(MODELS_DIR, 'PubLayNet_faster')}")
             print("    Asegúrate de que la ruta en 'main.py' coincida con tu carpeta real.")
        
        layout_config = LayoutConfig(
            config_path=config_path,
            model_path=model_path,
            label_map_name="PubLayNet",
            score_threshold=0.8,
        )
        # Inicializamos el pipeline de layout (que incluye preprocesamiento)
        self.layout_pipeline = DocumentLayoutPipeline(layout_config, ocr_languages=['spa'])

        # --- B. MOTOR OCR ---
        print("\n[2/3] 👁️  Cargando Modelos OCR (TrOCR + LaTeX)...")
        self.ocr_engine = ContentExtractor(use_gpu=self.use_gpu)

        # --- C. MOTOR RAG ---
        print("\n[3/3] 🧠 Conectando Base de Conocimiento (RAG)...")
        self.rag_engine = UltraFastMarkdownRAG(
            model_size="small", 
            persist_directory=CHROMA_DB_DIR
        )
        
        print("\n✅ SISTEMA LISTO PARA PROCESAR.")

    def run(self, image_filename):
        """
        Ejecuta todo el proceso: Imagen -> Layout -> Recortes -> Markdown -> RAG -> Chat
        """
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        if not os.path.exists(image_path):
            print(f"❌ Error: La imagen '{image_filename}' no existe en la carpeta 'images/'.")
            return

        print(f"\n" + "="*60)
        print(f"📂 PROCESANDO DOCUMENTO: {image_filename}")
        print("="*60)

        # ---------------------------------------------------------
        # PASO 1: Layout y Recorte
        # ---------------------------------------------------------
        print("\n🔹 PASO 1: Análisis de Estructura (Layout)")
        
        # Este método ya guarda los recortes en 'images_processed/' automáticamente
        # gracias a tu clase DocumentLayoutPipeline corregida
        layout_results = self.layout_pipeline.process_document(
            image_path=image_path,
            preprocess=True  # True para pizarras (aplica CLAHE)
        )
        
        # Recuperamos la lista de recortes generados desde el diccionario de resultados
        crop_paths = layout_results.get("crops", [])
        
        if not crop_paths:
            print("❌ No se detectaron regiones en la imagen.")
            return

        print(f"   -> Se generaron {len(crop_paths)} fragmentos.")
        
        # Ordenamos visualmente (de arriba a abajo)
        crop_paths.sort(key=extract_sort_key)

        # ---------------------------------------------------------
        # PASO 2: Transcripción (OCR)
        # ---------------------------------------------------------
        print("\n🔹 PASO 2: Transcripción Inteligente")
        
        base_name = os.path.splitext(image_filename)[0]
        markdown_content = f"# Nota Digitalizada: {base_name}\n\n"
        
        for i, crop_path in enumerate(crop_paths, 1):
            label = extract_label_from_filename(crop_path)
            print(f"   -> Procesando fragmento {i}/{len(crop_paths)} [{label}]...", end=" ", flush=True)
            
            try:
                # Cargar recorte
                img = Image.open(crop_path).convert("RGB")
                
                # Transcribir usando el OCR adecuado según el label
                # (Asegúrate de haber agregado 'procesar_imagen_ya_recortada' a ContentExtractor como vimos antes)
                texto = self.ocr_engine.procesar_imagen_ya_recortada(img, label)
                
                markdown_content += texto + "\n"
                print("✅")
            except Exception as e:
                print(f"❌ ({e})")
                markdown_content += f"\n> [Error leyendo fragmento {i}]\n"

        # Guardar Markdown Final
        md_filename = f"{base_name}.md"
        md_path = os.path.join(MARKDOWNS_DIR, md_filename)
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"   -> 📝 Archivo Markdown guardado en: markdowns/{md_filename}")

        # ---------------------------------------------------------
        # PASO 3: Indexación RAG
        # ---------------------------------------------------------
        print("\n🔹 PASO 3: Indexación en Memoria")
        self.rag_engine.index_markdown(md_path)
        
        # ---------------------------------------------------------
        # PASO 4: Chat Interactivo
        # ---------------------------------------------------------
        self.iniciar_chat()

    def iniciar_chat(self):
        print("\n" + "="*60)
        print("🤖 CHAT INTERACTIVO (RAG)")
        print("   Pregunta sobre el contenido de tu imagen.")
        print("   Escribe 'salir' para terminar.")
        print("="*60)
        
        # Modelo LLM a usar (debe estar en Ollama)
        LLM_MODEL = "mistral" 

        while True:
            try:
                question = input("\n📝 Tú: ").strip()
                if not question: continue
                if question.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta la próxima!")
                    break
                
                print("   (Pensando...)")
                # Llamada al RAG
                respuesta = self.rag_engine.answer_question(question, model_name=LLM_MODEL)
                
                print(f"\n🤖 IA:\n{respuesta}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# ==============================================================================
# EJECUCIÓN DEL SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    USAR_GPU = False  # Cambia a True si tienes CUDA
    
    # NOMBRE DE LA IMAGEN (Debe estar dentro de la carpeta 'images')
    # Cambia esto por el nombre del archivo que quieras probar
    
    IMAGEN_OBJETIVO = input("Ingrese el nombre de la imagen: ") 

    try:
        pipeline = FullStackPipeline(use_gpu=USAR_GPU)
        pipeline.run(IMAGEN_OBJETIVO)
        
    except Exception as e:
        print(f"\n❌ Error fatal en la ejecución:\n{e}")
