FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); SentenceTransformer('BAAI/bge-reranker-base')"

# Start Ollama server in the background and pull the model
# (Doing this during build bakes the model into the image)
RUN ollama serve & sleep 5 && ollama pull qwen3:8b

# Copy the rest of the application
COPY . .

# Ensure entrypoint is executable
RUN chmod +x /app/sandbox_entrypoint.sh

# Run the sandbox pipeline
ENTRYPOINT ["/app/sandbox_entrypoint.sh"]
