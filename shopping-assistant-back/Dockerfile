FROM python:3.10-slim

COPY shoppingassistant shoppingassistant
COPY requirements_prod.txt requirements.txt
COPY api api
COPY models models
COPY raw_data/test_images raw_data/test_images
COPY raw_data/articles_filtered.csv raw_data/articles_filtered.csv
COPY raw_data/articles.csv raw_data/articles.csv
COPY raw_data/images_filtered raw_data/images_filtered
COPY raw_data/transactions_filtered.csv raw_data/transactions_filtered.csv
COPY embeddings embeddings

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
#CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]
