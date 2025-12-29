import json
import os
import layoutparser as lp
from PIL import Image
from layout_config import LayoutConfig

class LayoutDetector:
    def __init__(self, config: LayoutConfig):
        self.config = config
        self.model = self._load_model()
        
    def _load_model(self):
        return lp.Detectron2LayoutModel(
            config_path=self.config.config_path,
            model_path=self.config.model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.score_threshold],
            label_map=self.config.label_map
        )
    
    def detect_layout(self, image_path, output_json_path, output_viz_path=None):
        """
        Args:
            image_path: Ruta de la imagen entrada
            output_json_path: Ruta completa donde guardar el json (inc. carpeta)
            output_viz_path: Ruta completa donde guardar la imagen visual (inc. carpeta)
        """
        # Cargar imagen
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path # Es un objeto PIL
            
        # Detectar
        layout = self.model.detect(img)
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
        
        # Guardar JSON en la carpeta correcta
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data_export, f, indent=4)
        
        # Guardar Visualización en la carpeta correcta
        if output_viz_path:
            os.makedirs(os.path.dirname(output_viz_path), exist_ok=True)
            viz = lp.draw_box(img, layout, box_width=3)
            viz.save(output_viz_path)
            print(f"Visualización guardada en: {output_viz_path}")
        
        return layout
