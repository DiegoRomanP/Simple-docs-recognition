"""
Módulo que orquesta el pipeline completo de procesamiento de documentos.
"""

from layout_config import LayoutConfig
from layout_detector import LayoutDetector
from bounding_box_processor import BoundingBoxProcessor
from ocr_processor import OCRProcessor
from image_preprocessor import ImagePreprocessor


class DocumentLayoutPipeline:
    """Clase principal que orquesta el pipeline completo de procesamiento de documentos"""

    def __init__(self, layout_config: LayoutConfig, ocr_languages=None):
        """
        Inicializa el pipeline de procesamiento

        Args:
            layout_config: Configuración del modelo de layout
            ocr_languages: Idiomas para OCR
        """
        self.detector = LayoutDetector(layout_config)
        self.bbox_processor = BoundingBoxProcessor()
        self.ocr_processor = OCRProcessor(languages=ocr_languages, use_tesseract=True)
        self.preprocessor = ImagePreprocessor()

    def process_document(
        self,
        image_path,
        preprocess=True,
        detect_layout_flag=True,
        crop_boxes=True,
        output_coords="coordenadas.json",
    ):
        """
        Procesa un documento completo

        Args:
            image_path: Ruta de la imagen
            preprocess: Si aplicar preprocesamiento
            detect_layout_flag: Si detectar layout
            crop_boxes: Si recortar las bounding boxes
            output_coords: Archivo de salida para coordenadas

        Returns:
            Dict con rutas de archivos generados
        """ 
        results = {}

        # Preprocesamiento
        if preprocess:
            print("Preprocesando imagen...")
            preprocessed_path = self.preprocessor.preprocesar_pizarra_para_layout(
                image_path
            )
            results["preprocessed_image"] = preprocessed_path
            image_to_process = preprocessed_path
        else:
            image_to_process = image_path

        # Detección de layout
        if detect_layout_flag:
            print("Detectando layout...")
            #si es que se quiere procesar la imagen original cambiar image_to_process por image_path
            layout = self.detector.detect_layout(image_to_process, output_coords)
            results["layout"] = layout
            results["coords_file"] = output_coords

        # Recorte de bounding boxes
        if crop_boxes and detect_layout_flag:
            print("Recortando bounding boxes...")
            self.bbox_processor.recortar_bounding_boxes(image_to_process, output_coords)
            results["cropped_boxes"] = True

        return results

    def detect_text_in_crops(self, crop_filenames):
        """
        Detecta texto en una lista de imágenes recortadas

        Args:
            crop_filenames: Lista de nombres de archivos

        Returns:
            Dict con resultados por imagen
        """
        results = {}

        for filename in crop_filenames:
            result = self.ocr_processor.detectar_texto(filename)
            results[filename] = {"has_text": len(result) > 0, "ocr_result": result}

            if results[filename]["has_text"]:
                print(f"Imagen {filename} tiene texto detectado")

        return results
