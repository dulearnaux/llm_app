FROM python:3.10

WORKDIR /app

ENV OPENAI_API_KEY="your_api_key"

#RUN git
#RUN git clone https://github.com/... .

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copying all files after running ollama.sh should prevent pulling ollama models
# each time source code is updated
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
