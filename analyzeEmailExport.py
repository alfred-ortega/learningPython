import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk
import re

# Download necessary NLTK resources
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess email text for analysis
    
    Args:
        text (str): Raw email body text
    
    Returns:
        str: Cleaned and processed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def analyze_emails(file_path):
    """
    Analyze emails and provide label suggestions
    
    Args:
        file_path (str): Path to Excel file containing emails
    
    Returns:
        DataFrame: Original emails with suggested labels
    """
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Ensure there's a column for email body
    if 'body' not in df.columns:
        raise ValueError("Excel file must contain a 'body' column")
    
    # Preprocess email bodies
    df['processed_body'] = df['body'].apply(preprocess_text)
    
    # Create TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['processed_body'])
    
    # Perform clustering to suggest labels
    n_clusters = min(5, len(df))  # Adaptive number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # Extract top keywords for each cluster
    def get_cluster_keywords(cluster):
        cluster_docs = df[df['cluster'] == cluster]['processed_body']
        cluster_vectorizer = TfidfVectorizer(max_features=5)
        cluster_tfidf = cluster_vectorizer.fit_transform(cluster_docs)
        feature_names = cluster_vectorizer.get_feature_names_out()
        
        # Get top keywords
        keywords = [feature_names[idx] for idx in cluster_tfidf.toarray().mean(axis=0).argsort()[-5:][::-1]]
        return ', '.join(keywords)
    
    # Generate label suggestions
    cluster_keywords = {cluster: get_cluster_keywords(cluster) 
                        for cluster in range(n_clusters)}
    
    # Add label suggestions to DataFrame
    df['suggested_label'] = df['cluster'].map(cluster_keywords)
    
    return df[['body', 'cluster', 'suggested_label']]

def main():
    # Example usage
    file_path = 'emails.xlsx'
    
    try:
        # Analyze emails
        results = analyze_emails(file_path)
        
        # Display results
        print(results)
        
        # Optionally save results
        results.to_excel('email_analysis_results.xlsx', index=False)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()