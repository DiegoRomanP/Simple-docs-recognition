import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pix2tex.cli import LatexOCR

class ContentExtractor:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Cargando modelo OCR en {self.device}...")

        # Modulo para extraer texto (TrOCR)
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.text_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)

        # Modelo para mátematicas Latex OCR
        try:
            self.math_model = LatexOCR()
        except Exception as e:
            print(f'No se cargó el modelo de latex ({e})')
            self.math_model = None
        
    def _ocr_texto(self, image_crop):
        """ Procesar el texto manuscrito con TrOCR"""
        pixel_values = self.processor(images=image_crop, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.text_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def _ocr_formula(self, image_crop):
        """Procesa formula con pix2tex"""
        if self.math_model:
            try:
                # LatexOCR espera una imagen PIL
                return f"$${self.math_model(image_crop)}$$"
            except: 
                return "[Error en fórmula]"
        return self._ocr_texto(image_crop)

    # --- NUEVO MÉTODO AGREGADO ---
    def procesar_imagen_ya_recortada(self, imagen_crop, label):
        """
        Procesa una imagen que ya ha sido recortada.
        Args:
            imagen_crop (PIL.Image): La imagen pequeña ya cargada en memoria.
            label (str): La etiqueta ('Title', 'Text', 'Equation', etc.) extraída del nombre del archivo.
        """
        # Limpiar label por si viene con extension (ej: "Title.jpg")
        label = label.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")

        # Router de modelos
        if label in ['Title', 'Text', 'List', 'TextRegion']:
            contenido = self._ocr_texto(imagen_crop)
            
            # Formateo simple para Markdown
            if label in ['Title', 'Title Region']:
                return f"# {contenido}\n"
            elif label == 'List':
                return f"- {contenido}\n"
            else: # Text
                return f"{contenido}\n"

        elif label in ['Table', 'Figure', 'Equation', 'MathsRegion']: 
            return self._ocr_formula(imagen_crop)
        
        else:
            return self._ocr_texto(imagen_crop)
