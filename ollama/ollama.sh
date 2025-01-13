#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

ollama pull llama3.2:1b
ollama pull llama3.1:8b

ollama run llama3.2:1b
ollama run llama3.2:8b

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 1
done
