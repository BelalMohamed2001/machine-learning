# LogisticRegression Model 

import pandas as pd #to read csv file
from sklearn.preprocessing import StandardScaler # to scale our data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # the model on dataset
from sklearn.model_selection import train_test_split # to train and split dataset
from sklearn.metrics import accuracy_score, classification_report
import pickle


#load the csv file
df=pd.read_csv("Modified_SQL_Dataset.csv")


# to show first 5 row
print(df.head())



# select independant and depandent variables
X=df["Query"]
Y=df["Label"]

#spilt the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

# Convert SQL queries into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)



# Make predictions on the test set
predictions = model.predict(X_test_tfidf)
p=model.predict


# Evaluate the model
accuracy=accuracy_score(Y_test,predictions)
report=classification_report(Y_test,predictions)


print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# make pickle file of our model
pickle.dump(model,open("model.pkl","wb"))

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)




