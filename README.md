# Retrieval Augmented Generation (RAG) with Llama

Basic intro to creating a RAG model for your data.


## Getting Started

Use Ollama to run our LLMs. 

`curl -fsSL https://ollama.com/install.sh | sh`

# Pull llama3.1 model
`ollama pull llama3.1:8b model`

# Install python dependencies

`pip install ipython scikit-learn langchain langchain_community langchain-openai langchain-ollama streamlit BeautifulSoup`

# Run app

`streamlit run main.py`

This should create a web app at http://localhost:8501






