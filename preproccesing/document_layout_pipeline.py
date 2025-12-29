import os
from .layout_detector import LayoutDetector
from .bounding_box_processor import BoundingBoxProcessor
from .ocr_processor import OCRProcessor
from .image_preprocessor import ImagePreprocessor

class DocumentLayoutPipeline:
    def __init__(self, layout_config, ocr_languages=None):
        self.detector = LayoutDetector(layout_config)
        self.bbox_processor = BoundingBoxProcessor()
        self.ocr_processor = OCRProcessor(languages=ocr_languages, use_tesseract=True)
        self.preprocessor = ImagePreprocessor()
        
        self.dir_images = "images_processed"
        self.dir_coords = "coordenadas_layout"
        
        os.makedirs(self.dir_images, exist_ok=True)
        os.makedirs(self.dir_coords, exist_ok=True)

    def process_document(self, image_path, preprocess=True):
        results = {}
        filename = os.path.basename(image_path)
        name_no_ext = os.path.splitext(filename)[0]

        # 1. Preprocesamiento (Solo para ayudar al detector)
        if preprocess:
            print("Preprocesando imagen para detección...")
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(self.dir_images, processed_filename)
            
            # Esta imagen se usará SOLO para detectar el layout
            image_for_detection_path = self.preprocessor.preprocesar_pizarra_para_layout(
                image_path, output_path=processed_path
            )
            results["preprocessed_image"] = image_for_detection_path
        else:
            image_for_detection_path = image_path

        # 2. Detección de Layout
        print("Detectando layout...")
        json_filename = f"{name_no_ext}_coords.json"
        json_path = os.path.join(self.dir_coords, json_filename)
        viz_filename = f"viz_{filename}"
        viz_path = os.path.join(self.dir_images, viz_filename)

        # Usamos la imagen PROCESADA (o la original si preprocess=False) para detectar
        layout = self.detector.detect_layout(
            image_for_detection_path, 
            output_json_path=json_path,
            output_viz_path=viz_path
        )
        results["layout"] = layout
        results["coords_file"] = json_path
        results["viz_file"] = viz_path

        # 3. Recorte
        print("Recortando bounding boxes de la imagen ORIGINAL...")
        # --- CORRECCIÓN CRÍTICA AQUÍ ---
        # Usamos 'image_path' (la ORIGINAL) en lugar de 'image_for_detection_path'
        crops = self.bbox_processor.recortar_bounding_boxes(
            image_path,  # <--- CAMBIO IMPORTANTE
            json_path,
            self.dir_images
        )
        results["crops"] = crops

        return results