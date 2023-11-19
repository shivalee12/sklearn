# test_app.py
from app import app, input_data
import pytest

inp = input_data()
print(inp)


def test_post_predict():
    # Sample payload for testing
    payload = {"data": inp}

    # Make a POST request to the predict endpoint
    response = app.test_client().post("/predict", json=payload)
    print(response.get_data(as_text=True))

    # Assert the status code is 200
    assert response.status_code == 200

    # Assert the predicted digit for each possible digit (0-9)
    assert response.get_json()["prediction"] == 0
    assert response.get_json()["prediction"] == 1
    assert response.get_json()["prediction"] == 2
    assert response.get_json()["prediction"] == 3
    assert response.get_json()["prediction"] == 4
    assert response.get_json()["prediction"] == 5
    assert response.get_json()["prediction"] == 6
    assert response.get_json()["prediction"] == 7
    assert response.get_json()["prediction"] == 8
    assert response.get_json()["prediction"] == 9

    assert response.status_code == 200
