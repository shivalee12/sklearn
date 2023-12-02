import unittest
import requests
import base64


class FlaskApiTests(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"

    def test_svm_predict(self):
        # Prepare your test image data
        with open("test_image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        response = requests.post(
            f"{self.BASE_URL}/predict/svm", json={"image": [encoded_string]}
        )
        self.assertEqual(response.status_code, 200)
        # Add more assertions based on the expected response

    def test_dt_predict(self):
        # Prepare your test image data
        with open("test_image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        response = requests.post(
            f"{self.BASE_URL}/predict/dt", json={"image": [encoded_string]}
        )
        self.assertEqual(response.status_code, 200)

    def test_lr_predict(self):
        # Prepare your test image data
        with open("test_image.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        response = requests.post(
            f"{self.BASE_URL}/predict/lr", json={"image": [encoded_string]}
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
