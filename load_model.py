import joblib
import json
import pandas as pd
import numpy as np


class load_model_regression:
    def __init__(self, model_path: str):
        self.model_pickle = model_path
        self.model = None

    def load_model(self):
        with open(self.model_pickle, "rb") as model_file:
            self.model = joblib.load(model_file)
        return self.model

    def predict(self, data: list[np.number]):
        new_value = np.array(data).reshape(1, -1)
        return self.model.predict(new_value)


class load_model_classification:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load_model(self):
        with open(self.model_path, "rb") as model_file:
            self.model = joblib.load(model_file)
        return self.model

    def prediction(self, new_value: list):
        new_value = np.array(new_value).reshape(1, -1)
        return self.model.predict(new_value)


if __name__ == "__main__":
    # new_value luas_tanah,luas_bangunan,banyak_kamar_tidur dan banyak_kamar_mandi
    new_value = [1100, 700, 5, 6, 1]
    model_regression = load_model_regression("model/best_regression_model.pkl")
    # load model
    model_regression.load_model()
    prediction = model_regression.predict(new_value)
    print(prediction)
