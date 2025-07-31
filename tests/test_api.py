import requests


def test_predict():
    """Test the predict endpoint with the example from the prompt."""
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "class_name" in data
    assert data["class_id"] in [0, 1, 2]
