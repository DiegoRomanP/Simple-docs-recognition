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
cd Simple-docs-recognition
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio

pip install markdown
# Instalar dependencias del proyecto
pip install -r requirements.txt 
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

```

### Si aún no se detecta detectron2 instalar lo siguiente:
```bash
pip install torch torchvision
pip install "layoutparser[detectron2]"
pip install 'git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)'
```
### Descargar modelos:
No se podrá ejecutar el programa sin descargar los modelos.
```bash
python descargar_modelos.py
```
### Tener en cuenta antes de ejecutar el pipeline:
El programa solo leera las imagenes que esten en la carpeta images/ 
por lo tanto es importante crear una carpea con el nombre "images" en la carpeta raiz del proyecto.
Puedes hacerlo con el siguiente comando:
```bash
mkdir images
```
Y luego coloca tus imagenes en la carpeta images/.


#### Configuracion de modelos:

Asegúrate de que los pesos de los modelos de detección (model_final.pth y config.yml) estén ubicados en la carpeta models/PubLayNet_faster/.1. Descargar modelos de PubLayNet y TrOCR:
con el archivo 
descargar_modelos.py 
**ojo solo funciona en linux**
### Ejecutar el pipeline:
```bash
python main.py
```
### Aspectos modificables:
Si es que al momento de ejecutar el pipeline los resultados no son los esperados, puedes modificar los valores de los parametros en el archivo main.py
Como el modelo de procesamiento en la linea 92 y 93:
Antes:
```python
config_path = os.path.join(MODELS_DIR, "PubLayNet_faster", "config.yml")
model_path = os.path.join(MODELS_DIR, "PubLayNet_faster", "model_final.pth")
```
Despues:
```python
config_path = os.path.join(MODELS_DIR, "HJDataset_faster", "config.yml")
model_path = os.path.join(MODELS_DIR, "HJDataset_faster", "model_final.pth")
```
los modelos se encuentran en la carpeta models/H

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
