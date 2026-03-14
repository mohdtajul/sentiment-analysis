from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load Models
svm_model = pickle.load(open("models/svm_model.pkl", "rb"))
svm_vectorizer = pickle.load(open("models/svm_vectorizer.pkl", "rb"))

logistic_model = pickle.load(open("models/logistic_model.pkl", "rb"))
logistic_vectorizer = pickle.load(open("models/logistic_vectorizer.pkl", "rb"))

nb_model = pickle.load(open("models/nb_model.pkl", "rb"))
nb_vectorizer = pickle.load(open("models/nb_vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    comment = request.form["comment"]
    model_name = request.form["model"]

    if model_name == "svm":
        vec = svm_vectorizer.transform([comment])
        prediction = svm_model.predict(vec)

    elif model_name == "logistic":
        vec = logistic_vectorizer.transform([comment])
        prediction = logistic_model.predict(vec)

    elif model_name == "nb":
        vec = nb_vectorizer.transform([comment])
        prediction = nb_model.predict(vec)

    else:
        return render_template("index.html", prediction="Model not found")

    pred = prediction[0]

    if pred == -1:
        result = "Negative 😡"
    elif pred == 0:
        result = "Neutral 😐"
    else:
        result = "Positive 😊"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)