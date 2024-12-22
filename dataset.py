import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([stemmer.stem(lemmatizer.lemmatize(word)) for word in text.split() if word not in stop_words])
    return text

def load_combined_data():
    # Load datasets
    amazon_reviews = pd.read_csv('Amazon Review Data Web Scrapping.csv')
    train_data = pd.read_csv('train_data.csv', header=None)
    test_data = pd.read_csv('test_data.csv', header=None)

    # Process Amazon Reviews
    amazon_reviews_filtered = amazon_reviews[amazon_reviews['Own_Rating'] != 'Neutral']
    amazon_reviews_filtered['label'] = amazon_reviews_filtered['Own_Rating'].map({'Positive': 1, 'Negative': 0})
    amazon_reviews_filtered = amazon_reviews_filtered[['Review_text', 'label']]
    amazon_reviews_filtered.columns = [0, 1]  # Rename columns to match Train/Test

    # Combine datasets
    combined_data = pd.concat([train_data, test_data, amazon_reviews_filtered], ignore_index=True)
    combined_data[0] = combined_data[0].astype(str).apply(preprocess_text)

    # Shuffle combined data
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save combined data with headers
    combined_data.to_csv('combined_data.csv', index=False, header=['0', '1'])

if __name__ == "__main__":
    load_combined_data()