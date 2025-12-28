"""
Módulo para procesar y recortar bounding boxes.
"""

import json
from PIL import Image


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
        if isinstance(  image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        with open(coords_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for i, item in enumerate(data, start=1):
            x1, y1, x2, y2 = item["bbox"]
            recorte = image.crop((x1, y1, x2, y2))
            recorte.save(f"recorte_{item['label']}{i}.jpg")
