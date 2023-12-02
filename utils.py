from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics, svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import Normalizer
from joblib import dump, load
from sklearn import datasets
import numpy as np
import os


def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    normalizer = Normalizer(norm="l2")  # Using L2 norm for unit normalization
    data_normalized = normalizer.fit_transform(data)
    return data_normalized


def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target
    # get_info(x)
    return x, y


def split_data(x, y, test_size, dev_size, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test_size + dev_size), shuffle=False, random_state=random_state
    )
    test_size_adjusted = test_size / (test_size + dev_size)
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size_adjusted,
        shuffle=False,
        random_state=random_state,
    )
    return X_train, X_test, X_dev, y_train, y_test, y_dev


def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC

    if model_type == "DecisionTree":
        clf = tree.DecisionTreeClassifier

    if model_type == "LogisticRegression":
        clf = LogisticRegression

    model = clf(**model_params)
    model.fit(x, y)
    return model


def predict_and_eval(model, X_test, y_test):
    """
    Predicts labels for test data using the given model and evaluates its performance.

    Parameters:
    model (object): The trained machine learning model.
    X_test (array-like): Test data features.
    y_test (array-like): True labels for the test data.

    Returns:
    None
    """
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)

    # Visualize the first 4 test samples and show their predicted digit value
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    plt.savefig("predicted_digits.png")
    # Print the classification report
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Plot the confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # If needed, rebuild classification report from confusion matrix
    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )
    return  # predicted  # metrics.accuracy_score(y_test, predicted)


def hyperparameter_tuning(
    X_train, y_train, X_dev, y_dev, list_of_all_param_combinations, model_type
):
    best_accuracy = 0
    best_hparams = None
    best_model = None
    for param_combination in list_of_all_param_combinations:
        model = train_model(X_train, y_train, param_combination, model_type)
        # Evaluate the model on the development data
        accuracy = sum(y_dev == model.predict(X_dev)) / len(y_dev)

        # Check if this set of hyperparameters gives a better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hparams = param_combination
            best_model_path = "./models/{}".format(model_type) + ".joblib"
            best_model = model
            # save the best_model
    dump(best_model, best_model_path)

    print("Model save at {}".format(best_model_path))

    return best_hparams, best_model_path, best_accuracy


def train_and_save_models(X_train, y_train, roll_no):
    solvers = ["liblinear", "newton-cg", "lbfgs", "sag", "saga"]
    models_directory = "./models"

    for solver in solvers:
        model = LogisticRegression(solver=solver, max_iter=1000)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Train the model on the entire training set
        model.fit(X_train, y_train)

        # Save the model
        model_filename = f"{roll_no}_lr_{solver}.joblib"
        model_path = os.path.join(models_directory, model_filename)
        dump(model, model_path)

        # Print the performance of the model
        print(
            f"Solver: {solver}, Mean Accuracy: {mean_score:.3f}, Std: {std_score:.3f}"
        )
        print(f"Model saved as {model_path}")
