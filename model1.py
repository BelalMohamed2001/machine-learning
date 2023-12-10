# MODEL NAIVE BAYSE

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the CSV file
df = pd.read_csv("Modified_SQL_Dataset.csv")

# Display the first 5 rows of the DataFrame
print(df.head())

# Select independent and dependent variables
X = df["Query"]
Y = df["Label"]

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert SQL queries into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes model (Multinomial Naive Bayes)
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, Y_train)

# Make predictions on the test set
predictions_nb = model_nb.predict(X_test_tfidf)

# Evaluate the Naive Bayes model
accuracy_nb = accuracy_score(Y_test, predictions_nb)
report_nb = classification_report(Y_test, predictions_nb)

print(f"Naive Bayes Accuracy: {accuracy_nb}")
print("Naive Bayes Classification Report:\n", report_nb)

# Save the trained Naive Bayes model and TF-IDF vectorizer using pickle
with open('model_nb.pkl', 'wb') as model_file:
    pickle.dump(model_nb, model_file)

with open('tfidf_vectorizer_nb.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
