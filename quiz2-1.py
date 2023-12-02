from sklearn import tree, svm
from joblib import dump, load


def train_model(x, y, model_params, model_type):
    if model_type == "svm":
        clf = svm.SVC
    elif model_type == "DecisionTree":
        clf = tree.DecisionTreeClassifier
    else:
        raise ValueError("Invalid model_type")
    model = clf(**model_params)
    model.fit(x, y)
    return model


def tune_hparams(
    X_train, Y_train, X_dev, y_dev, list_of_all_param_combinations, model_type
):
    best_accuracy = 0
    best_hparams = None
    best_model_path = None

    for param_combination in list_of_all_param_combinations:
        model = train_model(X_train, Y_train, param_combination, model_type)
        accuracy = sum(y_dev == model.predict(X_dev)) / len(y_dev)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hparams = param_combination
            best_model_path = f"./models/{model_type}.joblib"
            best_model = model
            dump(best_model, best_model_path)

    print(f"Model saved at {best_model_path}")

    return best_hparams, best_model_path, best_accuracy


# Define your data and other undefined variables here

# Your code to define x, y, test_size_array, dev_size_array, split_train_dev_test, preprocessing, classifier_param_dict


# Run experiments
def run_exp():
    for test_size in test_size_array:
        for dev_size in dev_size_array:
            # your data splitting logic here, which should define X_train, X_test, X_dev, y_train, y_test, y_dev
            # Data preprocessing
            X_train = preprocessing(X_train)
            X_test = preprocessing(X_test)
            X_dev = preprocessing(X_dev)
            for model_type in classifier_param_dict:
                h_param = classifier_param_dict[model_type]
                best_hparams, best_model_path, dev_accuracy = tune_hparams(
                    X_train, y_train, X_dev, y_dev, h_param, model_type
                )
                best_model = load(best_model_path)
                train_acc = sum(y_train == best_model.predict(X_train)) / len(y_train)
                test_acc = sum(y_test == best_model.predict(X_test)) / len(y_test)
                print(f"Test Results for model type = {model_type}")
                print(
                    f"test size = {test_size}, dev size = {dev_size}, train size = {1 - (test_size+dev_size)}, train_acc = {train_acc}, test_acc = {test_acc}, dev_acc = {dev_accuracy}"
                )


run_exp()
