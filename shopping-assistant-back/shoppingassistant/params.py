import os
import numpy as np

########################### Variables ###########################

MODEL_TARGET = os.environ.get("MODEL_TARGET")






########################### Constants ##############################

# Paths (relative to project root)
print("Current working directory:", os.getcwd())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, 'raw_data', 'images_filtered')
ARTICLES_CSV = os.path.join(BASE_DIR, 'raw_data', 'articles_filtered.csv')
CHROMA_DIR = os.path.join(BASE_DIR, 'embeddings', 'chroma')
COLLECTION_NAME = 'shoe_embeddings'

# Mapping from user-friendly names to dataset column names
naming_convention = {'gender': 'index_group_name', 'Subcategory': 'product_type_name'}
