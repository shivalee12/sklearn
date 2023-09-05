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
from utils import preprocess_data, split_data, train_model, predict_and_eval, hyperparameter_tuning


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.savefig("digits.png")

digits = datasets.load_digits()


# Split data into train and test subsets
X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(digits.data, digits.target, 0.3, 0.2)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# Train the model
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")

# Call the predict_and_eval function
predict_and_eval(model, X_test, y_test)

# Show the plots
plt.show()
plt.savefig('confusion_matrix.png')

# hyperparameter tuning

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
test_sizes = [0.1, 0.2, 0.3] 
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1 - test_size - dev_size
        X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=test_size, random_state=42)
        X_dev, X_test, Y_dev, Y_test = train_test_split(X_test, Y_test, test_size=dev_size / (dev_size + test_size), random_state=42)

        best_hparams, best_model, best_accuracy = hyperparameter_tuning(X_train, y_train, X_dev, y_dev, gamma_ranges, C_ranges)

        print(f"test_size={test_size} dev_size={dev_size} train_size={train_size} train_acc={best_accuracy} dev_acc={best_accuracy} test_acc={best_accuracy}")
        print(f"Best Hyperparameters: {best_hparams}\n")