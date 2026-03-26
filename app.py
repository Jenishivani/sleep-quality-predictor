from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    sleep = float(request.form['sleep'])
    stress = float(request.form['stress'])

    result = model.predict([[sleep, stress]])

    # Convert result to message
    if result[0] == 1:
        msg = "Good Sleep 😴"
    elif result[0] == 0:
        msg = "Poor Sleep 😵"
    else:
        msg = "Average Sleep 😐"

    return render_template("index.html", output=msg)

if __name__ == "__main__":
    app.run(debug=True)