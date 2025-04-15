!pip install openpyxl
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
file_path = "/content/fake news detection dataset.xlsx"
df = pd.read_excel(file_path)

df['text'] = df['Title'].fillna('') + " " + df['Content'].fillna('')
df['label'] = df['Label'].map({'Real': 1, 'Fake': 0})

df.dropna(subset=['text', 'label'], inplace=True)


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.lower().split() if word not in stop_words]
    return ' '.join(tokens)


df['cleaned_text'] = df['text'].apply(preprocess_text)


df = df[df['cleaned_text'].str.strip().astype(bool)]


X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)


joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully.")


def predict_news(text):
    cleaned_text = preprocess_text(text)
    transformed = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed)
    return "Real News" if prediction[0] == 1 else "Fake News"


print("\n=== Fake News Detection System ===")
print("Enter a news headline or article to check if it is real or fake.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter news text: ")
    if user_input.lower() == 'exit':
        print("Exiting the Fake News Detection System. Goodbye!")
        break
    try:
        result = predict_news(user_input)
        print(f"Prediction: {result}\n")
    except Exception as e:
        print(f"Error: {e}. Please try again.\n")
