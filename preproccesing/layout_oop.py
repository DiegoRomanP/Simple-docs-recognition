import json
from transformers.models.tapas.modeling_tapas import ProductIndexMap
import layoutparser as lp
import cv2
import easyocr
from PIL import Image
import pytesseract
import numpy as np


class LayoutConfig:
    """Clase para gestionar la configuración de modelos y mapas de etiquetas"""
    
    # Label maps para diferentes modelos
    LABEL_MAPS = {
        "HJDatasets": {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 
                       5: "Title", 6: "Subtitle", 7: "Other"},
        "PubLayNet": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        "PrimaLayout": {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 
                        4: "MathsRegion", 5: "SeparatorRegion", 6: "OtherRegion"},
        "NewspaperNavigator": {0: "Photograph", 1: "Illustration", 2: "Map", 
                               3: "Comics/Cartoon", 4: "Editorial Cartoon", 
                               5: "Headline", 6: "Advertisement"},
        "TableBank": {0: "Table"},
        "MFD": {1: "Equation"}
    }
    
    def __init__(self, config_path, model_path, label_map_name=None, custom_label_map=None, score_threshold=0.8):
        """
        Inicializa la configuración del modelo
        
        Args:
            config_path: Ruta al archivo de configuración del modelo
            model_path: Ruta al archivo del modelo
            label_map_name: Nombre del label map predefinido (opcional)
            custom_label_map: Label map personalizado (opcional)
            score_threshold: Umbral de confianza para las detecciones
        """
        self.config_path = config_path
        self.model_path = model_path
        self.score_threshold = score_threshold
        
        # Determinar el label map a usar
        if custom_label_map:
            self.label_map = custom_label_map
        elif label_map_name and label_map_name in self.LABEL_MAPS:
            self.label_map = self.LABEL_MAPS[label_map_name]
        else:
            self.label_map = self.LABEL_MAPS["PubLayNet"]


class LayoutDetector:
    """Clase para detectar layouts en imágenes usando modelos de Detectron2"""
    
    def __init__(self, config: LayoutConfig):
        """
        Inicializa el detector de layout
        
        Args:
            config: Objeto LayoutConfig con la configuración del modelo
        """
        self.config = config
        self.model = self._load_model()
        
    def _load_model(self):
        """Carga el modelo Detectron2"""
        return lp.Detectron2LayoutModel(
            config_path=self.config.config_path,
            model_path=self.config.model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.score_threshold],
            label_map=self.config.label_map
        )
    
    def detect_layout(self, image_path, output_filename="coordenadas.json"):
        """
        Detecta el layout en una imagen y guarda las coordenadas en JSON
        
        Args:
            image_path: Ruta o objeto Image de PIL
            output_filename: Nombre del archivo JSON de salida
            
        Returns:
            Lista de bloques detectados
        """
        # Cargar imagen
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path
            
        # Detectar layout
        layout = self.model.detect(img)
        
        # Ordenar coordenadas por posición vertical
        layout.sort(key=lambda b: b.coordinates[1])
        
        # Exportar datos
        data_export = []
        for block in layout:
            x1, y1, x2, y2 = block.coordinates
            item = {
                "label": block.type,
                "score": float(block.score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            }
            data_export.append(item)
        
        # Guardar JSON
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data_export, f, indent=4)
        
        print(f"Se exportaron {len(data_export)} coordenadas a {output_filename}")
        
        # Dibujar visualización
        viz = lp.draw_box(img, layout, box_width=3)
        viz.save("resultado_detectado.jpg")
        
        print(f"¡Listo! Revisa el archivo {output_filename} en tu carpeta.")
        
        return layout


class BoundingBoxProcessor:
    """Clase para procesar y recortar bounding boxes"""
    
    @staticmethod
    def recortar_bounding_boxes(image_path, coords_filename="coordenadas.json"):
        """
        Recorta las bounding boxes detectadas
        
        Args:
            image_path: Ruta de la imagen original
            coords_filename: Archivo JSON con las coordenadas
        """
        image = Image.open(image_path)
        
        with open(coords_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for i, item in enumerate(data, start=1):
            x1, y1, x2, y2 = item["bbox"]
            recorte = image.crop((x1, y1, x2, y2))
            recorte.save(f"recorte_{item['label']}{i}.jpg")


class OCRProcessor:
    """Clase para procesar texto mediante OCR"""
    
    def __init__(self, languages=None, gpu=False, use_tesseract=False, tesseract_lang='spa'):
        """
        Inicializa el procesador OCR
        
        Args:
            languages: Lista de idiomas para EasyOCR
            gpu: Si usar GPU con EasyOCR
            use_tesseract: Si usar Tesseract en lugar de EasyOCR
            tesseract_lang: Idioma para Tesseract
        """
        self.use_tesseract = use_tesseract
        
        if use_tesseract:
            self.ocr_agent = lp.TesseractAgent(languages=tesseract_lang)
        else:
            self.languages = languages or ['en']
            self.gpu = gpu
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
    
    def detectar_texto(self, image_path):
        """
        Detecta texto en una imagen
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Resultados del OCR
        """
        image = Image.open(image_path)
        
        if self.use_tesseract:
            result = self.ocr_agent.detect(image)
        else:
            result = self.reader.readtext(image)
        
        return result


class ImagePreprocessor:
    """Clase para preprocesar imágenes antes de la detección"""
    
    @staticmethod
    def preprocesar_pizarra_para_layout(image_path_or_array, debug_save=False):
        """
        Preprocesa una imagen de pizarra usando operación Top-Hat
        
        Args:
            image_path_or_array: Ruta de imagen o array numpy
            debug_save: Si guardar imágenes intermedias para debug
            
        Returns:
            Imagen PIL procesada
        """
        # 1. Cargar imagen
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path_or_array}")
        else:
            img = image_path_or_array

        # 2. Convertir a Escala de Grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug_save:
            cv2.imwrite("debug_1_gray.jpg", gray)

        # 3. Detección y Corrección de Polaridad (Pizarra Negra vs Blanca)
        mean_intensity = np.mean(gray)
        if mean_intensity < 127:
            gray_for_processing = cv2.bitwise_not(gray)
            print("Detectada pizarra oscura. Invirtiendo colores.")
        else:
            gray_for_processing = gray
            print("Detectada pizarra clara.")
        
        if debug_save:
            cv2.imwrite("debug_2_inverted.jpg", gray_for_processing)

        # --- NÚCLEO DEL PROCESAMIENTO: Operación Top-Hat ---

        # 4. Definir el "Kernel" (Elemento Estructurante)
        kernel_size = (10, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        # 5. Aplicar Morphological Top-Hat
        tophat = cv2.morphologyEx(gray_for_processing, cv2.MORPH_TOPHAT, kernel)
        
        if debug_save:
            cv2.imwrite("debug_3_tophat_raw.jpg", tophat)

        # 6. Normalización / Estiramiento de Contraste
        normalized = cv2.normalize(tophat, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        if debug_save:
            cv2.imwrite("debug_4_normalized.jpg", normalized)

        # 7. Binarización (Umbralización de Otsu)
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 8. Inversión Final
        final_output_cv = cv2.bitwise_not(binary)
        
        if debug_save:
            cv2.imwrite("debug_5_final.jpg", final_output_cv)
            
        final_pil = Image.fromarray(final_output_cv)

        return final_pil
    
    @staticmethod
    def funcion_contraste(image_path):
        """
        Aplica mejora de contraste usando espacio de color YCrCb
        
        Args:
            image_path: Ruta de la imagen
        """
        image = cv2.imread(image_path)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        y_channel_stretched = cv2.normalize(y_channel, None, 0, 255, cv2.NORM_MINMAX)
        contrast_stretched_ycrb = cv2.merge((y_channel_stretched, cr_channel, cb_channel))
        contrast_stretched = cv2.cvtColor(contrast_stretched_ycrb, cv2.COLOR_YCR_CB2BGR)
        
        cv2.imwrite('Contrast_stretched_image.jpg', contrast_stretched)
    
    @staticmethod
    def preprocesar_global(ruta_imagen, output_path="imagen_preprocesada.jpg"):
        """
        Preprocesamiento global avanzado para OCR
        
        Args:
            ruta_imagen: Ruta de la imagen a procesar
            output_path: Ruta de salida de la imagen procesada
            
        Returns:
            Ruta del archivo guardado
        """
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        
        # 1. Denoising avanzado (crucial para OCR)
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        # 2. Mejora de contraste adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # 3. Binarización adaptativa (mejor que global)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        img = cv2.bitwise_not(img)
        
        # Guardar imagen preprocesada
        cv2.imwrite(output_path, img)
        
        return output_path
    
    @staticmethod
    def imagen_despues_modelo(image_path, output_path='segmento_mejorado.png'):
        """
        Mejora la imagen después del modelo (upsampling + denoising)
        
        Args:
            image_path: Ruta de la imagen
            output_path: Ruta de salida
        """
        img = cv2.imread(image_path)
        segmento_mejorado = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        segmento_mejorado = cv2.medianBlur(segmento_mejorado, 3)
        cv2.imwrite(output_path, segmento_mejorado)


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
    
    def process_document(self, image_path, preprocess=True, detect_layout_flag=True, 
                        crop_boxes=True, output_coords="coordenadas.json"):
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
            preprocessed_path = self.preprocessor.preprocesar_global(image_path)
            results['preprocessed_image'] = preprocessed_path
            image_to_process = preprocessed_path
        else:
            image_to_process = image_path
        
        # Detección de layout
        if detect_layout_flag:
            print("Detectando layout...")
            layout = self.detector.detect_layout(image_to_process, output_coords)
            results['layout'] = layout
            results['coords_file'] = output_coords
        
        # Recorte de bounding boxes
        if crop_boxes and detect_layout_flag:
            print("Recortando bounding boxes...")
            self.bbox_processor.recortar_bounding_boxes(image_to_process, output_coords)
            results['cropped_boxes'] = True
        
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
            results[filename] = {
                'has_text': len(result) > 0,
                'ocr_result': result
            }
            
            if results[filename]['has_text']:
                print(f"Imagen {filename} tiene texto detectado")
        
        return results


def main():
    """Función principal de ejemplo"""
    
    # Configurar el modelo
    config = LayoutConfig(
        config_path="models/NewspaperNavigator_faster/config.yml",
        model_path="models/NewspaperNavigator_faster/model_final.pth",
        custom_label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        score_threshold=0.8
    )
    
    # Crear pipeline
    pipeline = DocumentLayoutPipeline(config, ocr_languages=['spa'])
    
    # Procesar documento
    image = "images/imagen1.jpeg"
    results = pipeline.process_document(
        image_path=image,
        preprocess=True,
        detect_layout_flag=True,
        crop_boxes=True,
        output_coords="coordenadas.json"
    )
    
    print("Procesamiento completado!")
    print(f"Resultados: {results.keys()}")


if __name__ == "__main__":
    main()
