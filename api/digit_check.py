from flask import Flask, request, jsonify
from utils import predict_and_eval

app = Flask(__name__)


# Define a route to check if two images have the same digit
@app.route("/check_digit", methods=["POST"])
def check_digit():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"message": "Two image files are required"}), 400

    image1 = request.files["image1"].read()
    image2 = request.files["image2"].read()

    digit1 = predict_and_eval(image1)
    digit2 = predict_and_eval(image2)

    is_same_digit = digit1 == digit2

    return jsonify({"is_same_digit": is_same_digit})
