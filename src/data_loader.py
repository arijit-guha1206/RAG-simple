from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_all_documents(data_str: str) -> List[Any]:
    """
    Load all documents from the given data string.

    """
    #use project root data folder
    data_path = Path(data_str).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    
    documents = []
    
    #PDF files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] found {len(pdf_files)} PDF files : {[str(f) for f in pdf_files]}")
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF documents from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF file {pdf_file}: {e}")
            
    return documents        
            
    #Text files
    #CSV files
    