"""
Paquete de preprocesamiento y detección de layout de documentos.

Este paquete contiene clases para:
- Configurar modelos de detección de layout
- Detectar layouts en documentos
- Procesar bounding boxes
- Realizar OCR en imágenes
- Preprocesar imágenes
- Orquestar pipelines completos de procesamiento
"""

from .layout_config import LayoutConfig
from .layout_detector import LayoutDetector
from .bounding_box_processor import BoundingBoxProcessor
from .ocr_processor import OCRProcessor
from .image_preprocessor import ImagePreprocessor
from .document_layout_pipeline import DocumentLayoutPipeline

__all__ = [
    'LayoutConfig',
    'LayoutDetector',
    'BoundingBoxProcessor',
    'OCRProcessor',
    'ImagePreprocessor',
    'DocumentLayoutPipeline'
]

__version__ = '1.0.0'
