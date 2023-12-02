import matplotlib.pyplot as plt
import pdb
import itertools
from utils import *

classifier_param_dict = {}

# SVM
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
h_param_svm = {"gamma": gamma_ranges, "C": C_ranges}
h_param_svm_comb = [
    {"gamma": gamma, "C": C} for gamma, C in itertools.product(gamma_ranges, C_ranges)
]
test_size_array = [0.2]
dev_size_array = [0.2]
# classifier_param_dict["svm"] = h_param_svm_comb

# Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_param_tree = {}
h_param_tree["max_depth"] = max_depth_list
h_param_tree_comb = [{"max_depth": max_depth} for max_depth in max_depth_list]

# classifier_param_dict["DecisionTree"] = h_param_tree_comb

# LR
solvers = ["liblinear", "newton-cg", "lbfgs", "sag", "saga"]
h_param_lr = {}
h_param_lr["solver"] = solvers
h_param_lr_comb = [{"solver": sol} for sol in solvers]
classifier_param_dict["LogisticRegression"] = h_param_lr_comb


x, y = read_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, x, y):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


def run_exp():
    for test_size in test_size_array:
        for dev_size in dev_size_array:
            X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(
                x, y, test_size=0.3, dev_size=0.1
            )

            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            for model_type in classifier_param_dict:
                if model_type == "LogisticRegression":
                    train_and_save_models(X_train, y_train, "B20ME071")
                h_param = classifier_param_dict[model_type]
                best_hparams, best_model_path, dev_accuracy = hyperparameter_tuning(
                    X_train, y_train, X_dev, y_dev, h_param, model_type
                )
                # Loading best model
                best_model = load(best_model_path)

                train_acc = sum(y_train == best_model.predict(X_train)) / len(y_train)
                test_acc = sum(y_test == best_model.predict(X_test)) / len(y_test)
                # predict_and_eval(best_model, X_test, y_test)
                print("Test Results for model type = ", model_type)
                print(
                    "test size = ",
                    test_size,
                    "dev size = ",
                    dev_size,
                    "train size = ",
                    1 - (test_size + dev_size),
                    "train_acc = ",
                    train_acc,
                    "test_acc = ",
                    test_acc,
                    "dev_acc = ",
                    dev_accuracy,
                )
                run_results = {
                    "model name": model_type,
                    "test size": test_size,
                    "dev size": dev_size,
                    "train size": 1 - (test_size + dev_size),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "dev_acc": dev_accuracy,
                }

    svm_model = load("models/svm.joblib")
    tree_model = load("models/DecisionTree.joblib")
    lr_model = load("models/LogisticRegression.joblib")
    svm_pred = svm_model.predict(X_test)
    tree_pred = tree_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(svm_pred, tree_pred)
    print("confusion matrix = \n", confusion_matrix)
    cnf2 = [
        [sum(svm_pred == y_test), sum(svm_pred != y_test)],
        [sum(tree_pred == y_test), sum(tree_pred != y_test)],
        [sum(lr_pred == y_test), sum(lr_pred != y_test)],
    ]
    print("confusion matrix 2 = ", cnf2)


run_exp()
