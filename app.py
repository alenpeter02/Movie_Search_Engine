import streamlit as st
import sqlite3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Connect to the SQLite database
def connect_db(database_path):
    conn = sqlite3.connect(database_path)
    return conn

# Function to retrieve top 10 unique names based on query
def compute_average_embeddings(conn, model):
    c = conn.cursor()
    movie_embeddings = {}
    c.execute("SELECT name, embedding FROM film_data")
    for row in c.fetchall():
        name, embedding = row
        embeddings = json.loads(embedding)
        embeddings = np.array(embeddings)
        # Compute the average embedding for each movie
        average_embedding = np.mean(embeddings, axis=0)
        movie_embeddings[name] = average_embedding
    return movie_embeddings

# Function to find the top 10 unique movie names based on query
def get_top_10_unique_names(query, movie_embeddings, model):
    query_embedding = model.encode([query])[0]
    similarities = []
    for name, embedding in movie_embeddings.items():
        similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
        similarities.append((name, similarity))
        
    sorted_names = [name for name, _ in sorted(similarities, key=lambda x: x[1], reverse=True)]
    unique_names = []
    for name in sorted_names:
        if name not in unique_names:
            unique_names.append(name)
            if len(unique_names) == 10:
                break
    
    return unique_names

# Main function to run the Streamlit app
def main():
    st.title('Movie Name Search Engine')
    st.header('Enter subtitle')
    query = st.text_input('Enter your query:')
    
    if st.button('Search'):
        if query:
            conn = connect_db('chromadb.db')
            model = SentenceTransformer('all-MiniLM-L6-v2')
            movie_embeddings = compute_average_embeddings(conn, model)
            top_10_unique_names = get_top_10_unique_names(query, movie_embeddings, model)
            st.write('Top 10 Unique Names:')
            for i, name in enumerate(top_10_unique_names, start=1):
                st.write(f"{i}. {name}")
            conn.close()

if __name__ == '__main__':
    main()