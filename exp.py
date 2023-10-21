"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import (
    preprocess_data,
    split_data,
    train_model,
    predict_and_eval,
    hyperparameter_tuning,
)
from skimage.transform import resize
from sklearn.metrics import accuracy_score


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.savefig("digits.png")

digits = datasets.load_digits()


# Split data into train and test subsets
X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(
    digits.data, digits.target, 0.3, 0.2
)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# Train the model
model = train_model(X_train, y_train, {"gamma": 0.001}, model_type="svm")

# Call the predict_and_eval function
predict_and_eval(model, X_test, y_test)

# Show the plots
plt.show()
plt.savefig("confusion_matrix.png")

# hyperparameter tuning

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

# for test_size in test_sizes:
#     for dev_size in dev_sizes:
#         train_size = 1 - test_size - dev_size
#         X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(
#             digits.data, digits.target, test_size, dev_size, random_state=1
#         )
#         # print("X_train shape:", X_train.shape)
#         # print("y_train shape:", y_train.shape)
#         # print("X_dev shape:", X_dev.shape)
#         # print("y_dev shape:", y_dev.shape)
#         best_model, optimal_gamma, optimal_C = hyperparameter_tuning(
#             X_train, y_train, X_dev, y_dev, gamma_ranges, C_ranges
#         )

#         train_acc = predict_and_eval(best_model, X_train, y_train)
#         dev_acc = predict_and_eval(best_model, X_dev, y_dev)
#         test_acc = predict_and_eval(best_model, X_test, y_test)

#         print(
#             f"test_size={test_size} dev_size={dev_size} train_size={train_size} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc}"
#         )
#         # print(f"Best Hyperparameters: {best_hparams}\n")

# Define the image sizes to evaluate
image_sizes = [4, 6, 8]
for size in image_sizes:
    print(f"Image size: {size}x{size}")

    # Resize the images to the specified size
    X_train_resized = [resize(image, (size, size)) for image in X_train]
    X_dev_resized = [resize(image, (size, size)) for image in X_dev]
    X_test_resized = [resize(image, (size, size)) for image in X_test]

    # Hyperparameter tuning
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]

    best_model, optimal_gamma, optimal_C = hyperparameter_tuning(
        X_train_resized, y_train, X_dev_resized, y_dev, gamma_ranges, C_ranges
    )

    # Train the model with the best hyperparameters
    model = train_model(
        X_train_resized,
        y_train,
        {"gamma": optimal_gamma, "C": optimal_C},
        model_type="svm",
    )

    # Evaluate the model on train, dev, and test sets
    train_predictions = predict_and_eval(model, X_train_resized, y_train)
    dev_predictions = predict_and_eval(model, X_dev_resized, y_dev)
    test_predictions = predict_and_eval(model, X_test_resized, y_test)

    # Calculate accuracy for each set
    train_acc = accuracy_score(y_train, train_predictions)
    dev_acc = accuracy_score(y_dev, dev_predictions)
    test_acc = accuracy_score(y_test, test_predictions)

    # Print the results
    print(
        f"Image size: {size}x{size} train_size: 0.7 dev_size: 0.1 test_size: 0.2 Train accuracy: {train_acc:.2f} dev_acc: {dev_acc:.2f} test_acc: {test_acc:.2f}"
    )

total_samples = len(X_train) + len(X_test) + len(X_dev)
print(
    f"The number of total samples in the dataset (train + test + dev): {total_samples}"
)
# print(X_train.shape)
# Get the shape of the first image in the dataset
first_image_shape = digits.images[0].shape

# Extract height and width from the shape
image_height, image_width = first_image_shape
print(
    f"Size (height and width) of the images in dataset: Height={image_height}, Width={image_width}"
)
