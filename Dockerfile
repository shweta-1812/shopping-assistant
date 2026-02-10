# Base image
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements.txt


# Install combined dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY shopping-assistant-back/shoppingassistant shopping-assistant-back/shoppingassistant
COPY shopping-assistant-back/api shopping-assistant-back/api

# Copy frontend
COPY shopping-assistant-front shopping-assistant-front

# Copy models, raw_data, embeddings
COPY shopping-assistant-back/models shopping-assistant-back/models
COPY shopping-assistant-back/raw_data shopping-assistant-back/raw_data
COPY shopping-assistant-back/embeddings shopping-assistant-back/embeddings

# Expose port for Streamlit (Cloud Run public port)
EXPOSE 8080

# Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# Start FastAPI in background & Streamlit in foreground
CMD uvicorn shopping-assistant-back.api.fast:app --host 0.0.0.0 --port 8000 & \
    streamlit run shopping-assistant-front/app.py --server.port 8080 --server.headless true
