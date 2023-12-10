# we will use CNN MODEL because it has the highest accuracy

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the pre-trained CNN model
with open('model_cnn.pkl', 'rb') as file:
    model = pickle.load(file)  # Use the correct filename for your saved Keras model

# Load the Tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the SQL query from the form input
        query = request.form['queryInput']

        # Tokenize and pad the query text
        query_seq = tokenizer.texts_to_sequences([query])
        query_padded = pad_sequences(query_seq, maxlen=model.input_shape[1], padding='post')

        # Make a prediction using the trained model
        prediction = model.predict(query_padded)

        # Interpret the prediction (adjust as needed based on your model output)
        predicted_class = 1 if prediction > 0.5 else 0

        # Return the result to index.html
        return render_template("index.html", prediction_text=f"The predicted label is {predicted_class}")

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
