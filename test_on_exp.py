import itertools
import joblib
import pytest
import unittest
from sklearn.linear_model import LogisticRegression
from joblib import load


def test_hparamcount():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    list_of_all_param_combinations = [
        {"gamma": gamma, "C": C}
        for gamma, C in itertools.product(gamma_ranges, C_ranges)
    ]
    assert len(C_ranges) * len(gamma_ranges) == len(list_of_all_param_combinations)


rollno = "B20ME071"
solvers = ["liblinear", "newton-cg", "lbfgs", "sag", "saga"]


def test_loaded_model_is_logistic_regression():
    for solver in solvers:
        model_path = f"models/{rollno}_lr_{solver}.joblib"
        model = load(model_path)
        assert isinstance(
            model, LogisticRegression
        ), f"Model in {model_path} is not a Logistic Regression model"


def test_solver_name_matches():
    for solver in solvers:
        model_path = f"models/{rollno}_lr_{solver}.joblib"
        model = load(model_path)
        model_solver = model.get_params()["solver"]
        assert (
            model_solver == solver
        ), f"Solver in model {model_path} is {model_solver}, expected {solver}"
