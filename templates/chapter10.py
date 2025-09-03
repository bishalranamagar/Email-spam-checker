from flask import Flask, request, render_template
import pickle
import os

# Load the model and vectorizer
try:
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    
    # Load the vectorizer and model
    with open(os.path.join(models_dir, "cv.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    
    with open(os.path.join(models_dir, "clf.pkl"), "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    tokenizer = None
    model = None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", text="", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if tokenizer is None or model is None:
        return "Model not loaded properly. Please check the model files.", 500
    
    try:
        # Get the email content from the form
        email_text = request.form.get("email-content", "")
        
        # Transform the text using the vectorizer
        tokenized_email = tokenizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(tokenized_email)
        
        # Convert prediction to 1 (spam) or -1 (not spam)
        # Assuming your model returns 1 for spam and -1 for not spam
        prediction_value = int(prediction[0])
        
        return render_template("index.html", 
                              text=email_text, 
                              prediction=prediction_value)
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)

