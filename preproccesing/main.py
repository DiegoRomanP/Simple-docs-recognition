"""
Script principal para ejecutar el pipeline de procesamiento de documentos.
"""

from layout_config import LayoutConfig
from document_layout_pipeline import DocumentLayoutPipeline
import os

def main():
    """Función principal de ejemplo para procesar documentos"""

    print("=" * 60)
    print("Pipeline de Procesamiento de Documentos")
    print("=" * 60)

    # ========================================
    # 1. Configurar el modelo
    # ========================================
    print("\n[1/3] Configurando modelo...")
    
    # Asegúrate de que estas rutas sean correctas en tu PC
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Ajusta esta ruta según donde tengas tus modelos realmente
    MODEL_DIR = "/home/diego/Documentos/proyecto_pam/models/PubLayNet_faster"
    
    config = LayoutConfig(
        config_path=os.path.join(MODEL_DIR, "config.yml"),
        model_path=os.path.join(MODEL_DIR, "model_final.pth"),
        label_map_name="PubLayNet",
        score_threshold=0.8,
    )

    print(f"✓ Modelo configurado")

    # ========================================
    # 2. Crear pipeline
    # ========================================
    print("\n[2/3] Inicializando pipeline...")

    pipeline = DocumentLayoutPipeline(config, ocr_languages=["spa"])

    print("✓ Pipeline inicializado")

    # ========================================
    # 3. Procesar documento
    # ========================================
    print("\n[3/3] Procesando documento...")

    image_path = "/home/diego/Documentos/proyecto_pam/images/imagen1.jpeg"

    if not os.path.exists(image_path):
        print(f"❌ Error: No se encuentra la imagen en {image_path}")
        return

    # --- CORRECCIÓN AQUÍ ---
    # Llamada simplificada según tu nueva clase DocumentLayoutPipeline
    results = pipeline.process_document(
        image_path=image_path,
        preprocess=True
    )

    # ========================================
    # 4. Mostrar resultados
    # ========================================
    print("\n" + "=" * 60)
    print("Resultados del Procesamiento")
    print("=" * 60)

    # Verificamos las llaves que devuelve tu nuevo pipeline
    if "preprocessed_image" in results:
        print(f"\n✓ Imagen preprocesada: {results['preprocessed_image']}")

    if "coords_file" in results:
        print(f"✓ Coordenadas guardadas en: {results['coords_file']}")
        
    if "viz_file" in results:
        print(f"✓ Visualización guardada en: {results['viz_file']}")

    if "crops" in results:
        print(f"✓ Se generaron {len(results['crops'])} recortes en 'images_processed/'")

    if "layout" in results:
        print(f"✓ Elementos detectados: {len(results['layout'])}")

    print("\n" + "=" * 60)
    print("¡Procesamiento completado!")
    print("=" * 60)

    return results

if __name__ == "__main__":
    main()
