from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
import pdb

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_data(x, y, test_size, dev_size, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test_size + dev_size), shuffle=False, random_state=random_state
    )
    test_size_adjusted = test_size / (test_size + dev_size)
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_temp, y_temp, test_size=test_size_adjusted, shuffle=False, random_state=random_state
    )
    return X_train, X_test, X_dev, y_train, y_test, y_dev

def train_model(x,y,model_params,model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    
    model = clf(**model_params)
    # pdb.set_trace()
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
