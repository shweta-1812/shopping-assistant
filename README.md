#### Project Description ####

This project builds a computer vision–based product similarity system that retrieves visually similar products from an image dataset.
The system fine-tunes pre-trained convolutional neural networks to classify products and generate image embeddings. 
These embeddings are then used in a retrieval pipeline to find visually similar products using vector similarity search.
The goal of the project was to explore representation learning and retrieval-based systems for visual search applications.

#### Key Features ####

Fine-tuning of pre-trained CNN models for product classification
Handling class imbalance using class weights during training
Generation of 512-dimensional image embeddings
Vector similarity search using cosine similarity
Retrieval pipeline to find visually similar products within categories

#### Tech Stack #####

Python
TensorFlow / Keras
Computer Vision (CNNs)
Vector similarity search
Docker
Google Cloud Run
