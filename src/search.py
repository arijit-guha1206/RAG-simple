import os
from dotenv import load_dotenv
from httpx2 import query
from langchain_core import chat_history
from src.vector_store import FaissVectorStore
from langchain_groq import ChatGroq


load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("groq_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    #def search_and_summarize(self, query: str, top_k: int = 5) -> str:
    #    results = self.vectorstore.query(query, top_k=top_k)
    #    texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
    #    context = "\n\n".join(texts)
    #   if not context:
    #        return "No relevant documents found."
    #    prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
    #    response = self.llm.invoke([prompt])
    #    return response.content
    
    
    def search_and_summarize(self, query: str, top_k: int = 5, chat_history: list = None) -> str:
        chat_history = chat_history or []
        #step 1 : understand the convo
        standalone_query = query
        if chat_history:
            history_text = "\n".join(f"{msg['role']}: {msg['content']}"
                                      for msg in chat_history[-6:])
            rewrite_prompt = f"""you are a helpful assistant.
            Read the convo history and the latest question.
            
            
            Conversation History:
            {history_text}
            Latest Question: 
            {query}
            
            
            Rewrite the latest question as a standalone question
            that can be understood without the convo history.
            
            If the question is already clear keep it unchanged.
            
            Return only the written question.
            """
            rewritten = self.llm.invoke(rewrite_prompt)
            standalone_query = rewritten.content.strip() 
            
        #step 2: search FAISS using improved questions
        results = self.vectorstore.query(standalone_query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."
        ## STEP 3: Generate an answer using retrieved context
        answer_prompt = f"""
                You are a helpful assistant answering questions
                using the provided documents.

                Conversation history:
                {history_text if chat_history else "No previous conversation."}

                User's question:
                {query}

                Relevant document information:
                {context}

                Instructions:
                - Answer using the provided document information.
                - Use conversation history to understand the question.
                - Do not invent facts.
                - If the documents don't contain the answer,
                clearly say that the information was not found.

                Answer:
                """

        response = self.llm.invoke(answer_prompt)

        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)