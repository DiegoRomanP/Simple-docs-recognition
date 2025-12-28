"""
Script principal para ejecutar el pipeline de procesamiento de documentos.

Este script demuestra cómo usar las diferentes clases del paquete preproccesing
para procesar documentos, detectar layouts y realizar OCR.
"""

from layout_config import LayoutConfig
from document_layout_pipeline import DocumentLayoutPipeline


def main():
    """Función principal de ejemplo para procesar documentos"""

    print("=" * 60)
    print("Pipeline de Procesamiento de Documentos")
    print("=" * 60)

    # ========================================
    # 1. Configurar el modelo
    # ========================================
    print("\n[1/3] Configurando modelo...")

    config = LayoutConfig(
        config_path="/home/diego/Documentos/proyecto_pam/models/PubLayNet_faster/config.yml",
        model_path="/home/diego/Documentos/proyecto_pam/models/PubLayNet_faster/model_final.pth",
        label_map_name="PubLayNet",
        score_threshold=0.8,
    )

    print(f"✓ Modelo configurado: {config.model_path}")
    print(f"✓ Umbral de confianza: {config.score_threshold}")
    print(f"✓ Etiquetas: {list(config.label_map.values())}")

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

    results = pipeline.process_document(
        image_path=image_path,
        preprocess=True,
        detect_layout_flag=True,
        crop_boxes=True,
        output_coords="coordenadas.json",
    )

    # ========================================
    # 4. Mostrar resultados
    # ========================================
    print("\n" + "=" * 60)
    print("Resultados del Procesamiento")
    print("=" * 60)

    if "preprocessed_image" in results:
        print(f"\n✓ Imagen preprocesada: {results['preprocessed_image']}")

    if "coords_file" in results:
        print(f"✓ Coordenadas guardadas en: {results['coords_file']}")

    if "cropped_boxes" in results:
        print("✓ Bounding boxes recortadas exitosamente")

    if "layout" in results:
        print(f"✓ Elementos detectados: {len(results['layout'])}")

    print("\n" + "=" * 60)
    print("¡Procesamiento completado!")
    print("=" * 60)

    return results


def example_custom_workflow():
    """
    Ejemplo de flujo de trabajo personalizado usando componentes individuales
    """
    from layout_config import LayoutConfig
    from layout_detector import LayoutDetector
    from image_preprocessor import ImagePreprocessor
    from bounding_box_processor import BoundingBoxProcessor

    print("\n" + "=" * 60)
    print("Ejemplo de Flujo de Trabajo Personalizado")
    print("=" * 60)

    # Configurar modelo
    config = LayoutConfig(
        config_path="../models/NewspaperNavigator_faster/config.yml",
        model_path="../models/NewspaperNavigator_faster/model_final.pth",
        label_map_name="PubLayNet",
        score_threshold=0.75,
    )

    # Crear detector
    detector = LayoutDetector(config)

    # Preprocesar imagen
    preprocessor = ImagePreprocessor()
    image_path = "../images/imagen1.jpeg"
    preprocessed_path = preprocessor.preprocesar_global(
        image_path, "custom_preprocessed.jpg"
    )

    print(f"✓ Imagen preprocesada: {preprocessed_path}")

    # Detectar layout
    layout = detector.detect_layout(preprocessed_path, "custom_coords.json")

    print(f"✓ Layout detectado con {len(layout)} elementos")

    # Recortar bounding boxes
    bbox_processor = BoundingBoxProcessor()
    bbox_processor.recortar_bounding_boxes(preprocessed_path, "custom_coords.json")

    print("✓ Bounding boxes recortadas")
    print("=" * 60)


if __name__ == "__main__":
    # Ejecutar pipeline principal
    results = main()

    # Descomentar para ejecutar el ejemplo de flujo personalizado
    # example_custom_workflow()
