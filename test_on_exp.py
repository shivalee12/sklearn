import itertools
import joblib
import pytest
import unittest
from sklearn.linear_model import LogisticRegression


def test_hparamcount():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    list_of_all_param_combinations = [
        {"gamma": gamma, "C": C}
        for gamma, C in itertools.product(gamma_ranges, C_ranges)
    ]
    assert len(C_ranges) * len(gamma_ranges) == len(list_of_all_param_combinations)


def test_loaded_model_is_logistic_regression(self):
    # Replace <rollno> and <solver_name> with actual values
    file_name = "<rollno>_lr_<solver_name>.joblib"
    loaded_model = joblib.load(file_name)

    self.assertIsInstance(loaded_model, LogisticRegression)
    self.assertEqual(loaded_model.__class__.__name__, "LogisticRegression")


def test_solver_name_match_in_file_name_and_model(self):
    # Replace <rollno> and <solver_name> with actual values
    file_name = "B20ME071_lr_LogisticRegression.joblib"
    loaded_model = joblib.load(file_name)

    # Extract solver name from the file name
    file_solver_name = file_name.split("_")[-1].split(".")[0]

    # Extract solver name from the loaded model
    model_solver_name = loaded_model.get_params()["solver"]

    self.assertEqual(file_solver_name, model_solver_name)
