import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', 'number', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load dataset FIRST
df = pd.read_csv("data/spam.csv")

# Then modify df
df["label"] = (df["label"] == "spam").astype(int)
df["text"] = df["text"].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2),max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(vectorizer.transform(X_test))

print("Naive Bayes Result:\n")
print(classification_report(y_test, y_pred_nb))

# Model 2: Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(vectorizer.transform(X_test))

print("\nLogistic Regression Result:\n")
print(classification_report(y_test, y_pred_lr))

# User Input Prediction
print("\n--- SMS Spam Detector ---")

while True:
    user_input = input("\nEnter a message (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    clean_input = preprocess(user_input)
    user_tfidf = vectorizer.transform([clean_input])
    prediction = nb_model.predict(user_tfidf)

    if prediction[0] == 1:
        print("Result: Spam 🚫")
    else:
        print("Result: Not Spam ✅")