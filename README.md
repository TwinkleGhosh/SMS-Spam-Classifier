# 📩 SMS Spam Classifier

## 🚀 Project Overview

The **SMS Spam Classifier** is a Machine Learning project that classifies SMS messages as **Spam 🚫** or **Not Spam ✅** using Natural Language Processing (NLP) techniques.

This project demonstrates how raw text data can be transformed into meaningful numerical features using **TF-IDF**, and then classified using machine learning algorithms like **Naive Bayes** and **Logistic Regression**.

---

## 🎯 Objectives

* Convert unstructured text into numerical features
* Build and compare classification models
* Understand core NLP concepts like TF-IDF
* Create an interactive system for real-time prediction

---

## 🧠 Key Concepts Used

* Text Preprocessing (cleaning, normalization)
* TF-IDF Vectorization
* N-grams (Unigrams + Bigrams)
* Supervised Machine Learning
* Model Evaluation Metrics

---

## 🛠️ Technologies & Libraries

* Python 🐍
* Pandas
* Scikit-learn
* Regular Expressions (re)

---

## 📂 Project Structure

```
SMS-Spam-Classifier/
│
├── data/
│   └── spam.csv
│
├── main.py   
│
├── README.md
├── requirements.txt

---

## ⚙️ How It Works

### 1. Data Loading

The dataset is loaded and labels are converted:

* Spam → 1
* Not Spam (Ham) → 0

---

### 2. Text Preprocessing

* Convert text to lowercase
* Remove punctuation
* Replace numbers with keyword `"number"`

Example:

```
"Congrats! You won 5000 rupees!!!"
→ "congrats you won number rupees"
```

---

### 3. Feature Extraction (TF-IDF)

Text is converted into numerical vectors using **TF-IDF (Term Frequency - Inverse Document Frequency)**.

* Removes common words (stopwords)
* Uses **bigrams** (e.g., "free money", "win prize")

---

### 4. Model Training

#### ✅ Naive Bayes

* Fast and efficient for text classification
* Works well with word frequencies

#### ✅ Logistic Regression

* More generalized model
* Used for comparison

---

### 5. Model Evaluation

Performance is evaluated using:

* Precision
* Recall
* F1-score
* Accuracy

---

### 6. Interactive Prediction

Users can enter custom SMS messages in the terminal:

```
Enter a message: Win a free iPhone now!
Result: Spam 🚫
```

---

## ▶️ How to Run the Project

### Step 1: Clone Repository

```
git clone https://github.com/TwinkleGhosh/SMS-Spam-Classifier.git
cd SMS-Spam-Classifier
```

---

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

---

### Step 3: Run the Program

```
python src/main.py
```

---

## 🧪 Example Test Cases

| Message                                | Prediction |
| -------------------------------------- | ---------- |
| "Congratulations! You won 5000 rupees" | Spam 🚫    |
| "Hey, are we meeting today?"           | Not Spam ✅ |
| "Free entry in contest now!!!"         | Spam 🚫    |

---

## 📊 Skills Demonstrated

* Natural Language Processing (NLP)
* Machine Learning Model Building
* Data Preprocessing
* Model Evaluation & Comparison
* Python Project Structuring

---

## 💡 Future Improvements

* Build a **Streamlit Web App**
* Deploy model online
* Add more advanced NLP techniques
* Improve accuracy with larger datasets

---

## 🏁 Conclusion

This project provides a strong foundation in **text classification and NLP pipelines**, which are widely used in real-world applications like spam filtering, sentiment analysis, and customer support automation.

---

## 👤 Author

Twinkle Ghosh

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub and share!
