import unittest
from app import app
import argparse
import os
import shutil
from typing import List, Union
import re
import sys
import time
from datetime import datetime
class BasicTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.client = app.test_client()
        cls.save_dir = datetime.now().strftime("%Y%m%d %H%M%S")
        os.mkdir(cls.save_dir)
    def verify_ml(self, models : Union[List[str], None], task :str, **kwargs):
        if models is None:
            if task == "MLRegression":
                models = ["SVR", "DecisionTreeRegressor", "KNeighborsRegressor", "LinearRegression"]
            elif task == "MLClassification":
                models = ["SVC", "DecisionTreeClassifier", "KNeighborsClassifier", "LogisticRegression"]
        files = os.listdir(self.save_dir)
        for model in models:
            pattern = f"^{model}.pkl$"
            flag = False
            for file in files:
                if re.match(pattern, file):
                    flag = True
            self.assertTrue(flag, f"File for model {model} does not exist")

    def test_regression(self):
        # TODO maybe create payloads in a json and load it??
        task = "MLRegression"
        for dataset in os.listdir(os.path.join("Datasets", "Regression")):
            dataset_path = os.path.join("Datasets", "Regression", dataset)
            response = self.client.post("ml_predict", json = {
                "name" : "Iris",
                "task" : "MLRegression",
                "dataset_path" : dataset_path,
                "target_column" : "price",
                "preprocessing" : True,
                "imputation" : False,
                "scaling" : False,
                "models" : None,
                "hyper_parameter_tuning" : 0,
                "categorical_columns" : [],
                "path_to_save" : self.save_dir,
                "reduce_features" : False,
                "handle_imbalanced_data" : "",
                "ignore_cols" : []
            })
            self.assertEqual(response.status_code, 200, f"Status code for regression for dataset {dataset} was not 200. Status Code: {response.status_code}")
            self.verify_ml(models=None, task='MLRegression')

    def tearDown(self):
        shutil.rmtree(self.save_dir)
        return super().tearDown()

    def __del__(self):
        try:
            shutil.rmtree(self.save_dir)
        except:
            pass