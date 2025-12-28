"""
Módulo para procesar texto mediante OCR.
"""

import easyocr
import layoutparser as lp
from PIL import Image


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
