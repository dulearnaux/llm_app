# Retrieval Augmented Generation (RAG) with Llama

Basic intro to creating a RAG model for your data.


## Getting Started

You can either launch with `docker-compose`, or manually launch it with
`streamlit`

### Docker

This should take about 5 minutes to build. Downloads about 8Gb for Ollama and
models.  
`docker-compose up`

Visit the app on:  
`http://localhost:8501`


Check that Ollama is running. Should say `Ollama is Running` 
`http://localhost:11434`

Check that the two models are pulled `llama3.1:8b` and `llama3.2:1b`  
http://localhost:11434/api/tags

Check that one the two models are running.  
http://localhost:11434/api/ps

#### Notes
* You might need `OPENAI_API_KEY` as env variable if you choose open AI
embeddings in the web app.
  * You have to set this in an .env file in the same dir as `compose.yml`
  * E.g. `echo 'OPENAI_API_KEY=your-key' > .env`
* `compose.yml` expects your machine to have a GPU. If you don't have a GPU,
delete the `deploy` block in `compose.yml`. It will be slow without a GPU.  

### Streamlit

Install and use Ollama to run the  LLMs locally.  
`curl -fsSL https://ollama.com/install.sh | sh`

Pull one or both of these models  
`ollama pull llama3.1:8b model`  
`ollama pull llama3.2:1b model`

Install python dependencies  
`pip install -r requirements.txt`

Run the app  
`streamlit run main.py`

This should create a web app at http://localhost:8501

