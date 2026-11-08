from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    #docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("when was paleolithic age?", top_k=3))
    
    
    rag_search = RAGSearch()
    query = "when was paleolithic age?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    