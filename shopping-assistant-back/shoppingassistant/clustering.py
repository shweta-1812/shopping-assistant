import os
import pandas as pd
from PIL import Image
from pathlib import Path

import open_clip
import torch
import chromadb
from shoppingassistant.params import *


# Module-level cache (loaded once, reused across calls)
_cache = {}


def _load_clip_model():
    """Load CLIP ViT-B/32 model and preprocessing (cached)."""
    if 'clip_model' not in _cache:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        model.eval()
        _cache['clip_model'] = model
        _cache['clip_preprocess'] = preprocess
    return _cache['clip_model'], _cache['clip_preprocess']


def _get_chroma_collection():
    """Get or create ChromaDB collection (cached)."""
    if 'collection' not in _cache:
        # Check if embeddings exist on disk
        if not os.path.exists(CHROMA_DIR) or len(os.listdir(CHROMA_DIR)) == 0:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            _cache['collection'] = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={'hnsw:space': 'cosine'}
            )
        else:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            _cache['collection'] = client.get_collection(name=COLLECTION_NAME)
    return _cache['collection']


def _load_articles():
    """Load articles metadata from CSV (cached)."""
    if 'articles' not in _cache:
        _cache['articles'] = pd.read_csv(ARTICLES_CSV)
    return _cache['articles']


def extract_embedding(image_path):
    """Extract a 512-D CLIP embedding from an image."""
    model, preprocess = _load_clip_model()
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
    return embedding.squeeze().numpy().astype(float)


def _filename_to_article_id(filename):
    """Convert filename like '0181160009.jpg' to integer article_id."""
    return int(filename.replace('.jpg', ''))


def _get_image_path(filename):
    """Resolve full image path: images_filtered/018/0181160009.jpg"""
    subfolder = filename[:3]
    return os.path.join(IMAGES_DIR, subfolder, filename)


def build_chroma_collection(batch_size=64, force=False):
    """Populate ChromaDB with CLIP embeddings for all filtered images.

    Reads articles_filtered.csv for metadata and processes all images
    in images_filtered/. Call this once from a notebook to build the DB.
        """
    collection = _get_chroma_collection()
    if not force and collection.count() > 0:
        print(f"ChromaDB collection already populated by {collection.count()}. Skipping build.")

    else:
        df = _load_articles()
        # Build lookup dict for metadata
        meta_lookup = {}
        for _, row in df.iterrows():
            meta_lookup[row['article_id']] = {
                'article_id': str(row['article_id']),
                'prod_name': str(row.get('prod_name', '')),
                'product_type_name': str(row.get('product_type_name', '')),
                'colour_group_name': str(row.get('colour_group_name', '')),
                'index_group_name': str(row.get('index_group_name', '')),
            }

        # Collect all image paths
        all_files = []
        for subfolder in sorted(os.listdir(IMAGES_DIR)):
            subfolder_path = os.path.join(IMAGES_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for fname in sorted(os.listdir(subfolder_path)):
                if fname.endswith('.jpg'):
                    all_files.append(fname)

        print(f"Found {len(all_files)} images to process")

        # Process in batches
        model, preprocess = _load_clip_model()

        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]

            ids = []
            embeddings = []
            metadatas = []

            for fname in batch_files:
                image_path = _get_image_path(fname)
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0)

                    with torch.no_grad():
                        emb = model.encode_image(img_tensor)
                    emb = emb.squeeze().numpy().astype(float)

                    article_id = _filename_to_article_id(fname)
                    meta = meta_lookup.get(article_id, {
                        'article_id': str(article_id),
                        'prod_name': '',
                        'product_type_name': '',
                        'colour_group_name': '',
                        'index_group_name': '',
                    })

                    ids.append(fname)
                    embeddings.append(emb.tolist())
                    metadatas.append(meta)
                except Exception as e:
                    print(f"  Skipping {fname}: {e}")

            if ids:
                collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
                print(f"  Processed {min(i + batch_size, len(all_files))}/{len(all_files)}")

        print(f"Done! Collection has {collection.count()} items.")
    return collection


def get_similar_items(image_path, n=5, subcategory=None, gender=None, batch_size=64):
    """
    Returns a list of n similar items based on the provided image URL and model.

    First iteration: Only consider the same subcategory and gender given from
    classification models.
    """
    collection = build_chroma_collection(batch_size=batch_size)

    # Extract CLIP embedding from query image
    query_embedding = extract_embedding(image_path)

    # Build ChromaDB where filter
    where_filter = None
    conditions = []
    if subcategory is not None:
        conditions.append({'product_type_name': subcategory})
    if gender is not None:
        conditions.append({'index_group_name': gender})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {'$and': conditions}

    # Query ChromaDB (returns results sorted by similarity)
    query_params = {
        'query_embeddings': [query_embedding.tolist()],
        'n_results': n,
        'include': ['metadatas', 'distances'],
    }
    if where_filter is not None:
        query_params['where'] = where_filter

    results = collection.query(**query_params)

    # Format output
    output = []
    for i in range(len(results['ids'][0])):
        filename = results['ids'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        similarity = 1.0 - distance  # ChromaDB cosine distance = 1 - similarity

        output.append({
            'article_id': int(meta.get('article_id', 0)),
            'filename': filename,
            'image_path': _get_image_path(filename),
            'similarity': similarity,
            'prod_name': meta.get('prod_name', ''),
            'product_type_name': meta.get('product_type_name', ''),
            'colour_group_name': meta.get('colour_group_name', ''),
            'index_group_name': meta.get('index_group_name', ''),
        })

    return output
