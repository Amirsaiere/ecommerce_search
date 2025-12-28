import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load Data
train = pd.read_csv('data/train.csv', encoding='ISO-8859-1')
descriptions = pd.read_csv('data/product_descriptions.csv')
attributes = pd.read_csv('data/attributes.csv')

# Data Preview
# print("=" * 40,"Train Dataset","=" * 40)
# print(train.head())
# print(train.info())
# print("=" * 40,"Description Dataset","=" * 40)
# print(descriptions.head())
# print(descriptions.info())
# print("=" * 40,"Attributes Dataset","=" * 40)
# print(attributes.head())
# print(attributes.info())

# Merge Train Dataset with Product Descriptions on UID
df = train.merge(descriptions, on='product_uid', how='left')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower() 
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['search_term_clean'] = df['search_term'].apply(preprocess_text)
df['product_title_clean'] = df['product_title'].apply(preprocess_text)
df['product_description_clean'] = df['product_description'].apply(preprocess_text)
df['combined_text'] = df['product_title_clean'] + ' ' + df['product_description_clean']

# Exploratory Search Analysis
## Query length analysis
df['query_length'] = df['search_term'].str.split().str.len()
df['title_length'] = df['product_title'].str.split().str.len()

print("\nQuery Length Statistics:")
print(df['query_length'].describe())

## Relevance distribution
print("\nRelevance Score Distribution:")
print(df['relevance'].value_counts().sort_index())

## Top queries
print("\nTop 10 Most Common Queries:")
print(df['search_term'].value_counts().head(10))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Query length distribution
axes[0, 0].hist(df['query_length'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Query Length (words)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Query Lengths')

# Plot 2: Relevance distribution
df['relevance'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_xlabel('Relevance Score')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Relevance Score Distribution')
axes[0, 1].tick_params(axis='x', rotation=0)

# Plot 3: Avg relevance by query length
avg_rel_by_length = df.groupby('query_length')['relevance'].mean()
axes[1, 0].plot(avg_rel_by_length.index, avg_rel_by_length.values, marker='o')
axes[1, 0].set_xlabel('Query Length')
axes[1, 0].set_ylabel('Average Relevance')
axes[1, 0].set_title('Average Relevance by Query Length')
axes[1, 0].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('search_query_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Analysis complete. Saved: search_query_analysis.png")

def main():
    pass


if __name__ == "__main__":
    main()
