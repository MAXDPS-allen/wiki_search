import transformers
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, UnidentifiedImageError
import json
import clip
import os
import streamlit as st
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT model and tokenizer
model_dir = './bert'  # Update to local directory
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.AutoModel.from_pretrained(model_dir)
model.to(device)

# Initialize CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.to(device)

# Define vector database path
vector_db_path = './release/vector_database_with_new_summarizer(2).json'
clip_vector_db_path = './release/clip_vector_database.json'
title_vector_db_path = './release/title_vector_database.json'
section_title_vector_db_path = './release/section_title_vector_database.json'
weight_model_path = './release/weight_model.pth'  # Path to save the trained weight model

def encode_text(text):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        outputs = outputs[0]
        embeddings = torch.mean(outputs, dim=1)
    return embeddings

def encode_long_text(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    if (num_tokens <= max_length):
        return encode_text(text)
    else:
        token_chunks = [tokens[i:i+max_length] for i in range(0, num_tokens, max_length)]
        embeddings = []
        weights = []
        for chunk in token_chunks:
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            chunk_embedding = encode_text(chunk_text)
            embeddings.append(chunk_embedding)
            weights.append(len(chunk))
        embeddings = torch.stack(embeddings).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        weighted_average = torch.sum(embeddings * weights, dim=0) / torch.sum(weights)
        weighted_average = torch.mean(weighted_average, dim=0).unsqueeze(0)
        return weighted_average

def update_vector_db(wikipedia_data, vector_db, title_vector_db, section_title_vector_db):
    for title, content in tqdm(wikipedia_data.items(), desc="Updating vector database"):
        # Update section_texts vectors
        if title not in vector_db:
            text = " ".join(content['section_texts'])
            # vector_db[title] = encode_long_text(summarize_text(text)).cpu().tolist()
            vector_db[title] = encode_long_text(text).cpu().tolist()
        
        # Update wiki/ title vectors
        wiki_title = title.split('/')[-1].replace('_', ' ')
        if title not in title_vector_db:
            title_vector_db[title] = encode_text(wiki_title).cpu().tolist()
        
        # Update section_titles vectors
        section_titles = " ".join(content['section_titles'])
        if title not in section_title_vector_db:
            section_title_vector_db[title] = encode_text(section_titles).cpu().tolist()

    save_vector_db(vector_db, vector_db_path)
    save_vector_db(title_vector_db, title_vector_db_path)
    save_vector_db(section_title_vector_db, section_title_vector_db_path)

def load_wikipedia_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        wikipedia_data = json.load(file)
    return wikipedia_data

def fetch_wikipedia_titles(wikipedia_data):
    return list(wikipedia_data.keys())

def sample_data(wikipedia_data, sample_percentage):
    total_entries = len(wikipedia_data)
    sample_size = max(1, int(total_entries * sample_percentage / 100))
    sampled_keys = list(wikipedia_data.keys())[:sample_size]
    return {key: wikipedia_data[key] for key in sampled_keys}

def save_vector_db(vector_db, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(vector_db, file)

def load_vector_db(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            vector_db = json.load(file)
    else:
        vector_db = {}
    return vector_db

def search_wikipedia(query, wikipedia_data, vector_db, title_vector_db, section_title_vector_db, top_k, text_type):
     # Check if query length exceeds 20 words
    if (len(word_tokenize(query)) > 20) and (text_type == "short"):
        # query = summarize_text(query)
        text_type = "long"
    elif (len(word_tokenize(query)) > 4):
        text_type = "middle"   
    
    query_embedding = encode_long_text(query).squeeze().to(device)

    titles = list(wikipedia_data.keys())
     
    # Load title, section_title, and section_text vectors from their databases
    title_embeddings = torch.stack([torch.tensor(title_vector_db[title]).squeeze() for title in titles]).to(device)
    section_title_embeddings = torch.stack([torch.tensor(section_title_vector_db[title]).squeeze() for title in titles]).to(device)
    section_text_embeddings = torch.stack([torch.tensor(vector_db[title]).squeeze() for title in titles]).to(device)

    # Calculate similarities for each level
    title_similarities = F.cosine_similarity(query_embedding, title_embeddings)
    section_title_similarities = F.cosine_similarity(query_embedding, section_title_embeddings)
    section_text_similarities = F.cosine_similarity(query_embedding, section_text_embeddings)

    # Combine similarities with different weights based on text type
    if text_type == "long":
        combined_similarities = title_similarities * 0 + section_title_similarities * 0 + section_text_similarities * 1
    elif text_type == "middle":
        combined_similarities = title_similarities * 1 + section_title_similarities * 1 + section_text_similarities * 0
    else :
        combined_similarities = title_similarities * 1 + section_title_similarities * 0 + section_text_similarities * 0     

    top_indices = torch.topk(combined_similarities, top_k).indices.tolist()
    return [titles[i] for i in top_indices]

def image_to_text_description(image, clip_vector_db, top_k):
    try:
        img = image.convert('RGB')
        img = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(img).cpu()
        text_features = torch.tensor(list(clip_vector_db.values()))
        
        similarities = F.cosine_similarity(image_features, text_features)
        top_matches_idx = similarities.topk(top_k).indices.tolist()
        
        best_descriptions = [list(clip_vector_db.keys())[idx] for idx in top_matches_idx]
        return best_descriptions
    except UnidentifiedImageError:
        st.error("Unidentified image file")
    return None

def search_image_query(wikipedia_data, query_image, clip_vector_db, top_k):
    descriptions = image_to_text_description(query_image, clip_vector_db, top_k)
    if descriptions is None:
        st.error("Query image could not be processed.")
        return []

    matching_titles = []
    for title, content in wikipedia_data.items():
        for description in descriptions:
            for full_description in content['image_reference_descriptions']:
                if description in full_description:
                    matching_titles.append(title)
                    break  # Stop searching within this article once a match is found
            if title in matching_titles:
                break  # Move to the next title once a match is found

    return matching_titles[:top_k]

def generate_descriptions(wikipedia_data):
    descriptions = []
    for entry in wikipedia_data.values():
        descriptions.extend([desc[:77] for desc in entry['image_reference_descriptions']])
    return descriptions

def encode_descriptions(descriptions, clip_vector_db):
    new_descriptions = [desc for desc in descriptions if desc not in clip_vector_db]
    if not new_descriptions:
        return clip_vector_db
 
    text_inputs = clip.tokenize(new_descriptions).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).cpu().tolist()
     
    for desc, feature in zip(new_descriptions, text_features):
        clip_vector_db[desc] = feature
    return clip_vector_db

def save_clip_vector_db(clip_vector_db, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(clip_vector_db, file)

def load_clip_vector_db(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            clip_vector_db = json.load(file)
    else:
        clip_vector_db = {}
    return clip_vector_db

def load_validation_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        validation_set = json.load(file)
    return validation_set

def validate_queries(validation_set, wikipedia_data, vector_db, clip_vector_db, title_vector_db, section_title_vector_db, model):
    results = {}
    for query_dict in validation_set:
        for key, url in query_dict.items():
            if key.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    query_image = Image.open(key)
                    top_image_links = search_image_query(wikipedia_data, query_image, clip_vector_db, top_k)
                    results[key] = {
                        "top_links": top_image_links,
                        "url": url
                    }
                except UnidentifiedImageError:
                    results[key] = {
                        "error": "Unidentified image file",
                        "url": url
                    }
            else:
                top_text_links = search_wikipedia(key, wikipedia_data, vector_db, title_vector_db, section_title_vector_db, top_k, model=model)
                results[key] = {
                    "top_links": top_text_links,
                    "url": url
                }
    return results


# Load Wikipedia data
wikipedia_data = load_wikipedia_data('./release/wiki_knowledge_base.json')

# Load or initialize vector databases
vector_db = load_vector_db(vector_db_path)
title_vector_db = load_vector_db(title_vector_db_path)
section_title_vector_db = load_vector_db(section_title_vector_db_path)

# Load or initialize CLIP vector database
clip_vector_db = load_clip_vector_db(clip_vector_db_path)

# Load validation set
validation_set = load_validation_set('./release/validation_set_1.json')

sample_wiki = sample_data(wikipedia_data, 100)  # Sample %

# Update vector database with any new entries from the Wikipedia data
update_vector_db(sample_wiki, vector_db, title_vector_db, section_title_vector_db)

# Generate descriptions and update CLIP vector database
descriptions = generate_descriptions(sample_wiki)
if not clip_vector_db:
    clip_vector_db = encode_descriptions(descriptions, clip_vector_db)
    save_clip_vector_db(clip_vector_db, clip_vector_db_path)

import streamlit as st
from PIL import Image, UnidentifiedImageError
import json

# Streamlit interface
st.title("Wikipedia Semantic Search")

# Common top_k selection
st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Select number of top results (top_k):", min_value=1, max_value=20, value=5)

# Text search
st.header("Text Search")
text_query = st.text_input("Enter text to search:")
if text_query:
    top_article_links = search_wikipedia(text_query, wikipedia_data, vector_db, title_vector_db, section_title_vector_db, top_k,'short')
    st.write("Top articles for text query:", top_article_links)

# Image search
st.header("Image Search")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    try:
        query_image = Image.open(uploaded_file)
        top_image_links = search_image_query(sample_wiki, query_image, vector_db, clip_vector_db, top_k=top_k)
        st.write("Top articles for image query:", top_image_links)
    except UnidentifiedImageError:
        st.error("Unidentified image file")

# Batch search
st.header("Batch Search")
uploaded_json = st.file_uploader("Choose a JSON file...", type="json")
if uploaded_json:
    results = {}
    data = json.load(uploaded_json)
    for query in data:
        if query.endswith(('.jpg', '.jpeg', '.png')):
            try:
                query_image = Image.open(query)
                top_image_links = search_image_query(sample_wiki, query_image, clip_vector_db, top_k=top_k)
                results[query] = {"top_links": top_image_links}
            except UnidentifiedImageError:
                results[query] = {"error": "Unidentified image file"}
        else:
            top_text_links = search_wikipedia(query, wikipedia_data, vector_db, title_vector_db, section_title_vector_db, top_k,'short')
            results[query] = {"top_links": top_text_links}

    st.download_button("Download results", data=json.dumps(results, indent=1), file_name="results.json")

# Validation set search
st.header("Validation Set Search")
uploaded_validation_set = st.file_uploader("Choose a JSON file for validation set...", type="json")
if uploaded_validation_set:
    validation_set = json.load(uploaded_validation_set)
    if st.button("Run Validation Search"):
        validation_results = validate_queries(validation_set, sample_wiki, vector_db, clip_vector_db, top_k=top_k)
        st.write("Validation set results:", validation_results)
        st.download_button("Download validation results", data=json.dumps(validation_results,indent=1), file_name="validation_results.json")
