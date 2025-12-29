"""
Sistema RAG local para PDFs con configuración flexible
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

# --- CORRECCIÓN: Imports actualizados a langchain_community ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # O langchain.text_splitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class PDFRAGSystem:
    """Sistema completo RAG para PDFs locales"""
    
    def __init__(
        self,
        model_name: str = "llama3:8b",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 1. Inicializar embeddings (local)
        print(f"Cargando embeddings: {embeddings_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Inicializar LLM local
        print(f"Inicializando LLM: {model_name}")
        self.llm = self._initialize_llm(model_name)
        
        # 3. Vector store
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
    
    def _initialize_llm(self, model_name: str):
        """Inicializa el LLM local según el tipo"""
        if ":" in model_name or model_name in ["llama3", "mistral", "codellama", "phi3"]:
            return Ollama(
                model=model_name,
                temperature=0.1,
                num_predict=512,
            )
        elif model_name.endswith(".gguf"):
            from langchain_community.llms import LlamaCpp
            return LlamaCpp(
                model_path=model_name,
                n_ctx=2048,
                temperature=0.1,
                verbose=False
            )
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")
    
    def load_and_process_pdf(self, pdf_path: str, clear_existing: bool = False) -> None:
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
            print("Cargando base vectorial existente...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # Solo añadir si hay nuevos chunks
            if chunks:
                self.vector_store.add_documents(chunks)
        else:
            print("Creando nueva base vectorial...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        # 4. Configurar retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # 5. Crear cadena QA
        self._create_qa_chain()
        print("✅ PDF procesado y listo para consultas")
    
    def _create_qa_chain(self):
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
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict:
        if not self.qa_chain:
            raise ValueError("Primero carga un PDF con load_and_process_pdf()")
        
        print(f"Pregunta: {question}")
        result = self.qa_chain.invoke({"query": question}) # .invoke es el método nuevo, pero __call__ sigue funcionando
        
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
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")