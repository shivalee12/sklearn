from flask import Flask, request, jsonify
from utils import predict_and_eval

app = Flask(__name__)


# Your prediction logic or model import
def predict_digit(input_digit):
    # Replace this with your actual prediction logic or model call
    predicted_digit = 0
    return predicted_digit


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/", methods=["POST"])
def hello_world_post():
    return {"op": "Hello, World POST " + request.json["suffix"]}


@app.route("/predict", methods=["POST"])
def predict():
    input_digit = request.json.get("input_digit", None)
    if input_digit is not None:
        predicted_digit = predict_and_eval(input_digit)
        return jsonify({"predicted_digit": predicted_digit})
    else:
        return jsonify({"error": "Input digit not provided"}), 400
