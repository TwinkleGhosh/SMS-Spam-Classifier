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
    
    # Normalize money symbols
    text = text.replace("₹", " rupees ")
    text = text.replace("rs", " rupees ")
    
    # Replace numbers
    text = re.sub(r'\d+', 'number', text)
    
    # Fix common spam tricks
    text = text.replace("0", "o")
    text = text.replace("1", "i")
    text = text.replace("@", "a")
    text = text.replace("$", "s")
    text = text.replace("3", "e")
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

# Load dataset 
df = pd.read_csv("data/spam.csv")

# Modified DF
df["label"] = (df["label"] == "spam").astype(int)
df["text"] = df["text"].apply(preprocess)

#Custom spam keywords
def add_flags(text):
    spam_keywords = ["win", "won", "free", "prize", "cash", "offer", "money", "rupees"]
    
    for word in spam_keywords:
        if word in text:
            text += " spamword spamword spamword"
    
    return text
df["text"] = df["text"].apply(add_flags)

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

#Adversarial Testing 
test_cases = [
    "Fr33 m0ney n0w!!!",
    "W!n c@sh pr!ze",
    "Congratulations, you won ₹5000",
    "FREE entry into contest!!!",
    "Hey bro, call me later"
]
model = nb_model
print("\n--- Adversarial Testing ---")
for msg in test_cases:
    clean = preprocess(msg)
    pred = model.predict(vectorizer.transform([clean]))
    print(f"{msg} --> {'Spam' if pred[0]==1 else 'Not Spam'}")

# User Input Prediction
print("\n--- SMS Spam Detector ---")

while True:
    user_input = input("\nEnter a message (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # Preprocess
    clean_input = preprocess(user_input)
    clean_input = add_flags(clean_input)

    # Vectorize
    user_tfidf = vectorizer.transform([clean_input])

    # Ensemble prediction
    nb_pred = nb_model.predict(user_tfidf)[0]
    lr_pred = lr_model.predict(user_tfidf)[0]

    final_pred = 1 if (nb_pred + lr_pred) >= 1 else 0

    # Optional probability (from LR)
    proba = lr_model.predict_proba(user_tfidf)[0][1]
    print(f"Spam Probability: {proba:.2f}")

    # Final result
    if final_pred == 1:
        print("Result: Spam 🚫")
    else:
        print("Result: Not Spam ✅")