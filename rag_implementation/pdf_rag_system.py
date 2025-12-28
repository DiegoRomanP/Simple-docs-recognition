"""
Sistema RAG local para PDFs con configuración flexible
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

# LangChain (versión ligera)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Para LLM local
from langchain_community.llms import Ollama  # Opción 1
# from langchain_community.llms import LlamaCpp  # Opción 2


class PDFRAGSystem:
    """Sistema completo RAG para PDFs locales"""
    
    def __init__(
        self,
        model_name: str = "llama3:8b",  # Ollama model name
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Inicializa el sistema RAG.
        
        Args:
            model_name: Nombre del modelo LLM (Ollama o ruta .gguf)
            embeddings_model: Modelo para embeddings
            persist_directory: Donde guardar la base vectorial
            chunk_size: Tamaño de fragmentos de texto
            chunk_overlap: Solapamiento entre fragmentos
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 1. Inicializar embeddings (local)
        print(f"Cargando embeddings: {embeddings_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},  # o 'cuda' si tienes GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Inicializar LLM local
        print(f"Inicializando LLM: {model_name}")
        self.llm = self._initialize_llm(model_name)
        
        # 3. Vector store (inicializar más tarde con documentos)
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
    
    def _initialize_llm(self, model_name: str):
        """Inicializa el LLM local según el tipo"""
        
        # Opción 1: Ollama (recomendado para facilidad)
        if ":" in model_name or model_name in ["llama3", "mistral", "codellama"]:
            return Ollama(
                model=model_name,
                temperature=0.1,  # Bajo para respuestas consistentes
                num_predict=512,  # Máximo tokens a generar
            )
        
        # Opción 2: Llama CPP (modelos .gguf)
        elif model_name.endswith(".gguf"):
            from langchain_community.llms import LlamaCpp
            return LlamaCpp(
                model_path=model_name,
                n_ctx=2048,  # Contexto máximo
                temperature=0.1,
                verbose=False
            )
        
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
    
    def load_and_process_pdf(
        self,
        pdf_path: str,
        clear_existing: bool = False
    ) -> None:
        """
        Carga y procesa un PDF para el RAG.
        
        Args:
            pdf_path: Ruta al archivo PDF
            clear_existing: Si True, borra la base existente
        """
        print(f"Procesando PDF: {pdf_path}")
        
        # 1. Cargar PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 2. Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"Documento dividido en {len(chunks)} chunks")
        
        # 3. Crear o cargar vector store
        if clear_existing and os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        
        if os.path.exists(self.persist_directory):
            # Cargar existente
            print("Cargando base vectorial existente...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # Añadir nuevos documentos
            self.vector_store.add_documents(chunks)
        else:
            # Crear nueva
            print("Creando nueva base vectorial...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        # 4. Configurar retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Traer 4 chunks más relevantes
        )
        
        # 5. Crear cadena QA
        self._create_qa_chain()
        
        print("✅ PDF procesado y listo para consultas")
    
    def _create_qa_chain(self):
        """Crea la cadena de pregunta-respuesta"""
        
        # Prompt personalizado para contexto de PDF
        prompt_template = """
        Eres un asistente especializado en analizar documentos PDF.
        Usa el siguiente contexto del PDF para responder la pregunta.
        Si no sabes la respuesta, di que no lo sabes, no inventes.
        
        Contexto del PDF:
        {context}
        
        Pregunta: {question}
        
        Respuesta detallada:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Crear cadena de QA
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict:
        """
        Haz una pregunta sobre el contenido del PDF.
        
        Args:
            question: Pregunta a realizar
            
        Returns:
            Dict con respuesta y fuentes
        """
        if not self.qa_chain:
            raise ValueError("Primero carga un PDF con load_and_process_pdf()")
        
        print(f"Pregunta: {question}")
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "N/A")
                }
                for doc in result["source_documents"]
            ]
        }
    
    def chat_session(self):
        """Inicia una sesión interactiva de chat"""
        print("\n" + "="*50)
        print("Chat con tu PDF. Escribe 'salir' para terminar.")
        print("="*50)
        
        while True:
            try:
                question = input("\n📝 Tu pregunta: ").strip()
                
                if question.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if not question:
                    continue
                
                respuesta = self.ask_question(question)
                
                print("\n🤖 Respuesta:")
                print(respuesta["answer"])
                
                if respuesta["sources"]:
                    print("\n📚 Fuentes utilizadas:")
                    for i, source in enumerate(respuesta["sources"], 1):
                        print(f"{i}. Página {source['page']}: {source['content']}")
                        
            except KeyboardInterrupt:
                print("\n\nInterrumpido por usuario")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
