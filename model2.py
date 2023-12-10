# CNN MODEL

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

# Tokenize the text data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text data to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to have the same length
max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

# Build a simple CNN model
model_cnn = Sequential()
model_cnn.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length))
model_cnn.add(Conv1D(128, 5, activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(1, activation='sigmoid'))

# Compile the model
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
model_cnn.fit(X_train_padded, Y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the CNN model
accuracy_cnn = model_cnn.evaluate(X_test_padded, Y_test)[1]
print(f"CNN Accuracy: {accuracy_cnn}")

# Save the trained CNN model and tokenizer using pickle
with open('model_cnn.pkl', 'wb') as model_file:
    pickle.dump(model_cnn, model_file)

with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
