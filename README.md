this is just a simple RAG pipeline made for ingesting PDF files till now. Later versions, will be able to parse not only PDF format data but also SQL, CSV, Text, excel, etc. format of data.


# RAGYT

A Streamlit chatbot is then added that answers questions using information retrieved from your documents.

## Setup

1. Install the project dependencies with `uv sync`.
2. Create a local `.env` file and add your Groq API key.
3. Start the app with `uv run streamlit run chatbot.py`.

Keep your `.env` file and private documents out of Git.
