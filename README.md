# ProyectoPAM: Digitalización Inteligente de Notas Manuscritas con Agentes IA

## Descripción del Proyecto

Este proyecto se enfoca en la **digitalización y estructuración inteligente de notas manuscritas**, transformando el contenido de pizarras y cuadernos en formatos digitales organizados y editables. Utilizando una combinación de tecnologías de Reconocimiento Óptico de Caracteres (OCR), Reconocimiento de Texto Manuscrito (HTR) y Agentes de Inteligencia Artificial (IA) basados en Large Language Models (LLMs), buscamos crear un sistema que no solo transcriba el texto, sino que también comprenda su contexto, lo estructure y lo integre en bases de conocimiento.

El objetivo principal es facilitar la gestión del conocimiento para estudiantes e investigadores, permitiendo una interacción más eficiente con sus apuntes y materiales de estudio.

## Características Principales

* **Digitalización Avanzada**: Conversión de imágenes de texto manuscrito (pizarras, cuadernos) a texto digital.
* **Comparación OCR vs. HTR**: Implementación y análisis de diferentes enfoques para el reconocimiento de texto, destacando las ventajas del HTR para la escritura a mano.
* **Agentes de IA para Estructuración del Conocimiento**: Utilización de LLMs para:
  * **Tokenización y Embeddings**: Procesamiento del texto para su comprensión semántica.
  * **Modelos Secuenciales (RNNs, Transformers)**: Aplicación de arquitecturas avanzadas para el análisis contextual.
  * **Generación Aumentada por Recuperación (RAG)**: Mejora de la precisión y relevancia de la información generada por los LLMs mediante la consulta de bases de conocimiento externas.
  * **Estructuración Automática**: Transformación del texto plano en formatos estructurados como Markdown o LaTeX, identificando secciones, títulos, listas y conceptos clave.
* **Integración y Usabilidad**: Diseño de un flujo de trabajo intuitivo para el usuario, desde la captura de la imagen hasta la obtención de la nota estructurada.

## Estructura del Repositorio
```
├── docs/ # Documentación del proyecto (informes, papers, etc.)
├── data/ # Conjuntos de datos de ejemplo (imágenes de pizarras, notas manuscritas)
├── src/ # Código fuente del proyecto
│ ├── ocr_module/ # Módulo para el reconocimiento OCR
│ ├── htr_module/ # Módulo para el reconocimiento HTR
│ ├── llm_agent/ # Módulo para los agentes IA y LLMs
│ │ ├── rag_system/ # Implementación del sistema RAG
│ │ └── knowledge_structuring/ # Lógica para la estructuración del conocimiento
│ ├── utils/ # Utilidades y funciones auxiliares
│ └── main.py # Script principal de ejecución
├── notebooks/ # Jupyter notebooks para experimentación y análisis
├── models/ # Modelos pre-entrenados o entrenados localmente
├── tests/ # Pruebas unitarias e integración
├── .gitignore # Archivos y directorios a ignorar por Git
├── README.md # Este archivo
├── requirements.txt # Dependencias del proyecto
└── LICENSE # Licencia del proyecto
```
