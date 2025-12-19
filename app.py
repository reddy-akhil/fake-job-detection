from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    color = ""

    if request.method == "POST":
        job_text = request.form["job_text"]
        cleaned = clean_text(job_text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]

        if prediction == "fake":
            result = "🚨 FAKE JOB POSTING"
            color = "red"
        else:
            result = "✅ AUTHENTIC JOB POSTING"
            color = "green"

    return render_template("index.html", result=result, color=color)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

