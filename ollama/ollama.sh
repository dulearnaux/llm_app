#!/bin/bash

echo "Starting Ollama server..."
ollama serve &
#SERVE_PID=$!

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 1
done

ollama pull llama3.2:1b
ollama pull llama3.1:8b

#wait $SERVE_PID