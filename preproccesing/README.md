# Preprocessing Package - Detección de Layout de Documentos

Este paquete proporciona herramientas para el preprocesamiento, detección de layout y OCR en documentos e imágenes.

## 📁 Estructura del Paquete

```
preproccesing/
├── __init__.py                      # Inicialización del paquete
├── layout_config.py                 # Configuración de modelos
├── layout_detector.py               # Detección de layouts
├── bounding_box_processor.py        # Procesamiento de bounding boxes
├── ocr_processor.py                 # Procesamiento OCR
├── image_preprocessor.py            # Preprocesamiento de imágenes
├── document_layout_pipeline.py      # Orquestador principal
├── main.py                          # Script de ejecución principal
└── README.md                        # Este archivo
```

## 🚀 Inicio Rápido

### Uso Básico con Pipeline Completo

```python
from layout_config import LayoutConfig
from document_layout_pipeline import DocumentLayoutPipeline

# Configurar modelo
config = LayoutConfig(
    config_path="../models/NewspaperNavigator_faster/config.yml",
    model_path="../models/NewspaperNavigator_faster/model_final.pth",
    custom_label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    score_threshold=0.8
)

# Crear pipeline
pipeline = DocumentLayoutPipeline(config, ocr_languages=['spa'])

# Procesar documento
results = pipeline.process_document(
    image_path="../images/imagen1.jpeg",
    preprocess=True,
    detect_layout_flag=True,
    crop_boxes=True,
    output_coords="coordenadas.json"
)
```

### Ejecutar desde la Línea de Comandos

```bash
cd preproccesing
python main.py
```

## 📚 Componentes del Paquete

### 1. LayoutConfig
Gestiona la configuración de modelos y mapas de etiquetas.

```python
from layout_config import LayoutConfig

# Usando un label map predefinido
config = LayoutConfig(
    config_path="path/to/config.yml",
    model_path="path/to/model.pth",
    label_map_name="PubLayNet",
    score_threshold=0.8
)

# Usando un label map personalizado
config = LayoutConfig(
    config_path="path/to/config.yml",
    model_path="path/to/model.pth",
    custom_label_map={0: "Text", 1: "Title"},
    score_threshold=0.75
)
```

**Label Maps Disponibles:**
- `HJDatasets`: Page Frame, Row, Title Region, Text Region, Title, Subtitle, Other
- `PubLayNet`: Text, Title, List, Table, Figure
- `PrimaLayout`: TextRegion, ImageRegion, TableRegion, MathsRegion, SeparatorRegion, OtherRegion
- `NewspaperNavigator`: Photograph, Illustration, Map, Comics/Cartoon, Editorial Cartoon, Headline, Advertisement
- `TableBank`: Table
- `MFD`: Equation

### 2. LayoutDetector
Detecta layouts en imágenes usando modelos Detectron2.

```python
from layout_detector import LayoutDetector

detector = LayoutDetector(config)
layout = detector.detect_layout("imagen.jpg", "coordenadas.json")
```

### 3. BoundingBoxProcessor
Procesa y recorta bounding boxes detectadas.

```python
from bounding_box_processor import BoundingBoxProcessor

BoundingBoxProcessor.recortar_bounding_boxes("imagen.jpg", "coordenadas.json")
```

### 4. OCRProcessor
Realiza OCR en imágenes usando EasyOCR o Tesseract.

```python
from ocr_processor import OCRProcessor

# Usar EasyOCR
ocr = OCRProcessor(languages=['es', 'en'], gpu=False)

# Usar Tesseract
ocr = OCRProcessor(use_tesseract=True, tesseract_lang='spa')

resultado = ocr.detectar_texto("imagen.jpg")
```

### 5. ImagePreprocessor
Preprocesa imágenes antes de la detección.

```python
from image_preprocessor import ImagePreprocessor

# Preprocesamiento global para OCR
preprocessor = ImagePreprocessor()
processed = preprocessor.preprocesar_global("imagen.jpg", "output.jpg")

# Preprocesamiento específico para pizarras
pizarra = preprocessor.preprocesar_pizarra_para_layout("pizarra.jpg", debug_save=True)

# Mejora de contraste
preprocessor.funcion_contraste("imagen.jpg")
```

### 6. DocumentLayoutPipeline
Orquesta el pipeline completo de procesamiento.

```python
from document_layout_pipeline import DocumentLayoutPipeline

pipeline = DocumentLayoutPipeline(config, ocr_languages=['spa'])

# Procesar documento completo
results = pipeline.process_document(
    image_path="imagen.jpg",
    preprocess=True,
    detect_layout_flag=True,
    crop_boxes=True
)

# Detectar texto en recortes
crop_files = ["recorte_Title1.jpg", "recorte_Text1.jpg"]
text_results = pipeline.detect_text_in_crops(crop_files)
```

## 🔧 Uso Avanzado

### Flujo de Trabajo Personalizado

Puedes usar los componentes individualmente para crear flujos personalizados:

```python
from layout_config import LayoutConfig
from layout_detector import LayoutDetector
from image_preprocessor import ImagePreprocessor
from bounding_box_processor import BoundingBoxProcessor

# 1. Configurar
config = LayoutConfig(
    config_path="models/config.yml",
    model_path="models/model.pth",
    label_map_name="PubLayNet"
)

# 2. Preprocesar
preprocessor = ImagePreprocessor()
processed = preprocessor.preprocesar_global("imagen.jpg", "preprocessed.jpg")

# 3. Detectar
detector = LayoutDetector(config)
layout = detector.detect_layout(processed, "coords.json")

# 4. Recortar
BoundingBoxProcessor.recortar_bounding_boxes(processed, "coords.json")
```

## 📋 Requisitos

```
layoutparser
detectron2
opencv-python (cv2)
easyocr
pytesseract
Pillow
numpy
transformers
```

## 💡 Ejemplos de Uso

Ver `main.py` para ejemplos completos de:
- Pipeline automático completo
- Flujo de trabajo personalizado con componentes individuales
- Configuración de diferentes modelos y parámetros

## 🐛 Depuración

Para habilitar el modo debug en el preprocesamiento de pizarras:

```python
preprocessor = ImagePreprocessor()
result = preprocessor.preprocesar_pizarra_para_layout(
    "imagen.jpg", 
    debug_save=True  # Guarda imágenes intermedias
)
```

Esto generará archivos:
- `debug_1_gray.jpg`: Imagen en escala de grises
- `debug_2_inverted.jpg`: Imagen invertida (si es necesario)
- `debug_3_tophat_raw.jpg`: Resultado de Top-Hat
- `debug_4_normalized.jpg`: Imagen normalizada
- `debug_5_final.jpg`: Resultado final

## 📝 Notas

- El paquete está optimizado para procesamiento de documentos, periódicos y pizarras
- Los métodos de preprocesamiento son estáticos y pueden usarse sin instanciar las clases
- El pipeline completo maneja automáticamente todo el flujo de trabajo

## 🤝 Contribución

Para añadir nuevos label maps, edita `layout_config.py` y añade tu configuración al diccionario `LABEL_MAPS`.
