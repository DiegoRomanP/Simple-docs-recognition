"""
Módulo para detectar layouts en imágenes usando modelos de Detectron2.
"""

import json
import layoutparser as lp
from PIL import Image

from layout_config import LayoutConfig


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
