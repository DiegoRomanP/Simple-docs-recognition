"""
Módulo para gestionar la configuración de modelos de detección de layout.
"""


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
