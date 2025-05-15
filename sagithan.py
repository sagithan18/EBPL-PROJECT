import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download nltk resources if not present
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Improved sample dataset with more examples
data = {
    "text": [
        # Fake news examples
        "Scientists confirm earth is flat after new study",
        "Aliens have landed in New York City last night",
        "Celebrity caught in scandalous affair with politician",
        "COVID-19 vaccine contains microchips to track people",
        "Man travels through time using secret government machine",
        "Election results were rigged, sources reveal",
        "Miracle cure for cancer discovered in the Amazon jungle",
        "Government controls weather using secret technology",
        "Fake news website claims the moon is made of cheese",
        "Eating chocolate daily helps you lose 10 pounds",

        # Real news examples
        "NASA discovers new planet in the habitable zone",
        "Local community comes together to clean the park",
        "New study shows benefits of exercise on mental health",
        "Economy grows by 3% in the last quarter",
        "Scientists publish research on climate change effects",
        "Government announces new education policy reforms",
        "New technology improves solar panel efficiency",
        "International peace talks lead to historic agreement",
        "Stock markets close at record highs today",
        "Researchers develop vaccine for seasonal flu"
    ],
    "label": [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1]  # 0=fake, 1=real
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess text data
df['cleaned_text'] = df['text'].apply(clean_text)

# Prepare features and labels
X = df['cleaned_text']
y = df['label']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_vec = vectorizer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Function to predict new input text
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    return label

if __name__ == "__main__":
    print("\n--- Fake News Detector ---")
    while True:
        inp = input("Enter news text (or 'exit' to quit): ")
        if inp.lower() == 'exit':
            break
        result = predict_news(inp)
        print(f"Prediction: {result}\n")
