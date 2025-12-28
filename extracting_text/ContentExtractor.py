import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pix2tex.cli import LatexOCR


class ContentExtractor:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Cargando modelo OCR en {self.device}...")

        # Modulo para extraer texto (TrOCR)
        # Usamos "handwritten" de Microsoft. 
        # Se cambiará a trocr-small-handwritten si es muy lento
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.text_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)

        # Modelo para mátematicas Latex OCR
        # Convierte imagen -> código latex
        try:
            self.math_model = LatexOCR()
        except Exception as e:
            print(f'No se cargo el modelo de latex ({e}')
            self.math_model = None
        
    def _ocr_texto(self, image_crop):
        """ Procesar el texto manuscrito con TrOCR"""
        #Preprocesamiento específico de TrOCR
        pixel_values = self.processor(images = image_crop, return_tensors = 'pt').pixel_values.to(self.device)
        
        #generacion
        generated_ids = self.text_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def _ocr_formula(self, image_crop):
        """Procesa formula con pix2tex"""
        if self.math_model:
            try:
                #LatexOCR espera una imagen PIL
                return f"$${self.math_model(image_crop)}$$"
            except: 
                return "[Error en fórmula]"
        return self._ocr_texto(image_crop) #fallback a texto normal
    
    def procesar_region(self, imagen_original, bbox, label):
        """
        Método principial que decide qué modelo usar según el label
        Args:
            imagen_original (PIL.Image): La imagen completa procesada
            bbox (list): [x1, y1, x2, y2]
            label (str): 'Text', 'Title', 'List', 'Equation', etc.

        """
        crop = imagen_original.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # 2. Enrutamiento (Router)
        if label in ['Title', 'Text', 'List']:
            contenido = self._ocr_texto(crop)
            
            # Formateo simple para Markdown según el tipo
            if label == 'Title':
                return f"# {contenido}\n"
            elif label == 'List':
                return f"- {contenido}\n"
            else: # Text
                return f"{contenido}\n"

        elif label in ['Table', 'Figure']: 
            # A veces las tablas o figuras se detectan como 'Equation' en algunos modelos, 
            # o si tienes un detector específico de fórmulas, úsalo aquí.
            # Si detectas que es una fórmula matemática:
            return self._ocr_formula(crop)
        
        else:
            return self._ocr_texto(crop)
