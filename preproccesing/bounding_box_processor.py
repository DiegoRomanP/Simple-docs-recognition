import json
import os
from PIL import Image

class BoundingBoxProcessor:
    
    @staticmethod
    def recortar_bounding_boxes(image_path, json_path, output_dir):
        """
        Args:
            image_path: Ruta imagen original/procesada
            json_path: Ruta del JSON de coordenadas
            output_dir: Carpeta donde guardar los recortes (images_processed)
        """
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # Crear carpeta si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener nombre base para los archivos (ej: "imagen1")
        base_name = os.path.splitext(os.path.basename(image_path if isinstance(image_path, str) else "imagen"))[0]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        saved_files = []
        for i, item in enumerate(data, start=1):
            x1, y1, x2, y2 = item["bbox"]
            label = item['label']
            
            # Recortar
            try:
                recorte = image.crop((x1, y1, x2, y2))
                
                # Nombre: images_processed/imagen1_crop_1_Title.jpg
                filename = f"{base_name}_crop_{i}_{label}.jpg"
                save_path = os.path.join(output_dir, filename)
                
                recorte.save(save_path)
                saved_files.append(save_path)
            except Exception as e:
                print(f"Error recortando bbox {i}: {e}")

        return saved_files
