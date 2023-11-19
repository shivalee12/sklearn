from flask import Flask, request, jsonify
from utils import predict_and_eval
from sklearn import datasets, svm
from utils import preprocess_data, split_data, train_model, predict_and_eval

digits = datasets.load_digits()
X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(
    digits.data, digits.target, 0.3, 0.2
)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)


def input_data():
    return X_test[0][0]


model = svm.SVC
# Train the model
model = train_model(X_train, y_train, {"gamma": 0.001}, model_type="svm")

app = Flask(__name__)


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
        predicted_digit = predict_and_eval(model, input_digit)
        return jsonify({"predicted_digit": predicted_digit})
    else:
        return jsonify({"error": "Input digit not provided"}), 400
