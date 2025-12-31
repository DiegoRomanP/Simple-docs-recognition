# ProyectoPAM: Digitalización Inteligente de Notas Manuscritas con Agentes IA

## Descripción del Proyecto

Este proyecto se enfoca en la **digitalización y estructuración inteligente de notas manuscritas**, transformando el contenido de pizarras y cuadernos en formatos digitales organizados y editables. 

El sistema implementa un pipeline completo que combina:
1. **Visión Computacional**: Detecta la estructura visual de la nota (Títulos, Texto, Figuras) usando modelos de *Layout Parsing*.
2. **HTR & OCR Híbrido**: Transcribe texto manuscrito con **TrOCR** y convierte fórmulas matemáticas complejas directamente a código **LaTeX**.
3. **RAG (Retrieval-Augmented Generation)**: Indexa el contenido generado en una base de datos vectorial para permitir consultas inteligentes mediante un Chatbot local.

El objetivo principal es facilitar la gestión del conocimiento para estudiantes e investigadores, permitiendo "chatear" con sus apuntes de clase.

## Características Principales

* **Detección de Layout**: Segmentación inteligente de la imagen usando modelos `Detectron2` (PubLayNet) para separar texto, títulos y gráficos.
* **OCR Especializado**:
  * **TrOCR (Microsoft)**: Para texto manuscrito general.
  * **LaTeX-OCR (Pix2Tex)**: Para ecuaciones matemáticas complejas.
* **Preprocesamiento Adaptativo**: Algoritmos de visión (CLAHE) para corregir iluminación y contraste en fotos de pizarras reales.
* **Sistema RAG Local**:
  * **Indexación**: Almacenamiento persistente en `ChromaDB`.
  * **Chat Interactivo**: Interfaz de preguntas y respuestas potenciada por **Ollama (Mistral/Llama3)**.
* **Estructuración Automática**: Generación de archivos Markdown (`.md`) limpios y organizados automáticamente.

## Requisitos Previos (Ollama)

Para la fase de chat y generación de respuestas, es necesario tener instalado Ollama corriendo localmente.

1. **Instalar Ollama:**
   ```bash
   curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
   ```
   
2. **Instalar modelos:**
   ```bash
   ollama pull mistral
   ```
### Crear entorno virtual y ejecutar el código:
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio
pip install markdown
# Instalar dependencias del proyecto
pip install -r requirements.txt 
pip install 'git+https://github.com/facebookresearch/detectron2.git'

#si aún no se detecta detectron2 instalar lo siguiente
pip install torch torchvision
pip install "layoutparser[detectron2]"
pip install 'git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)'
```
#### Configuracion de modelos:
Asegúrate de que los pesos de los modelos de detección (model_final.pth y config.yml) estén ubicados en la carpeta models/PubLayNet_faster/.1. Descargar modelos de PubLayNet y TrOCR:
con el archivo 
descargar_modelos.py 
**ojo solo funciona en linux**
### Ejecutar el pipeline:
```bash
python main.py
```

## Estructura del proyecto
```plaintext
├── images/                  # Carpeta de ENTRADA (coloca aquí tus fotos)
├── images_processed/        # Carpeta generada con recortes y visualizaciones
├── markdowns/               # Carpeta de SALIDA (notas digitalizadas .md)
├── models/                  # Pesos de modelos (Detectron2, etc.)
├── chroma_db/               # Base de datos vectorial persistente (RAG)
├── src/                     # Código fuente modular
│   ├── preproccesing/       # Módulo de Layout y Visión
│   │   ├── layout_detector.py
│   │   ├── image_preprocessor.py
│   │   └── ...
│   ├── extracting_text/     # Módulo de OCR y HTR
│   │   ├── ContentExtractor.py
│   │   └── ...
│   └── rag_implementation/  # Módulo de Inteligencia Artificial (RAG)
│       ├── ultra_fast_markdown_rag.py
│       └── ...
├── main_pipeline.py         # SCRIPT PRINCIPAL (Orquestador)
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Este archivo
└── LICENSE                  # Licencia del proyecto
```
