import streamlit as st
from src.search import RAGSearch


#page configuration
st.set_page_config(
    page_title = "My RAG Chatbot",
    page_icon = "🤖",
)


#chatbot heading
st.title("My RAG Chatbot")
st.write("Ask me anything about your uploaded documents!")


#Load RAG engine
@st.cache_resource
def load_rag():
    return RAGSearch()


rag_search = load_rag()

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
#Accept user's question 
if prompt := st.chat_input("Ask here...."):
    #display user msg
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
        })
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("thinkin...wait up!..."):
            
            try:
                answer = rag_search.search_and_summarize(
                    prompt,
                    top_k=3,
                    chat_history=st.session_state.messages[:-1]
                )
            except Exception as e:
                answer = f"nuh uh! something went wrong: {str(e)}"
                
        st.write(answer)
        
    #save AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
        })
