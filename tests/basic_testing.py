import unittest
from app import app
import argparse
import os
import shutil
from typing import List, Union
import re
import sys
import time
import shutil
import gdown
import json

INDENT_MULTIPLIER = 4

from datetime import datetime
class BasicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.save_dir = datetime.now().strftime("%Y%m%d %H%M%S")
        cls.log_file_name = f"testing_logs_{cls.save_dir}.log"
        os.makedirs(cls.save_dir, exist_ok=True)
        cls.empty_directory(cls.save_dir)
        cls.log("Start Test", indent=INDENT_MULTIPLIER * 0)
        try:
            shutil.rmtree('Datasets')
        except:
            pass
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1fz8aVuMXFe65DaaIyoSBsFoTVuYeiG7K?usp=drive_link",
            output='Datasets',
            quiet=False,
            use_cookies=False
        )
        assert os.path.exists("Datasets"), "Datasets folder not downloaded tests cannot be run exiting"

    @staticmethod
    def empty_directory(path):
        """
        Deletes all files and folders inside the given directory,
        but leaves the directory itself intact.
        """
        if not os.path.isdir(path):
            return

        for item in os.listdir(path):
            item_path = os.path.join(path, item)

            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) 

    @classmethod
    def log(cls, message, indent = 0):
        with open(cls.log_file_name, 'a') as f:
            message = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") + ": " + " "*indent + message
            f.writelines(message)
            f.writelines("\n")


    def verify_ml(self, models : Union[List[str], None], task :str, input_payload, **kwargs):
        self.__class__.log(f'Start verify_ml for task {task}, Payload used: {input_payload}', indent= INDENT_MULTIPLIER * 4)
        if models is None:
            if task == "MLRegression":
                models = ["DecisionTreeRegressor", "KNeighborsRegressor", "LinearRegression"]
            elif task == "MLClassification":
                models = ["DecisionTreeClassifier", "KNeighborsClassifier", "LogisticRegression"]
        self.__class__.log(f"Running verify_ml for models: {models}",indent= INDENT_MULTIPLIER * 4)
        files = os.listdir(self.save_dir)
        for model in models:
            self.__class__.log(f"Start verification for {model}.", indent= INDENT_MULTIPLIER * 5)
            pattern = f"^{model}.pkl$"
            flag = False
            for file in files:
                if re.match(pattern, file):
                    flag = True
            if flag:
                self.__class__.log(f"{model} verified.", indent= INDENT_MULTIPLIER * 5)
            else:
                self.__class__.log(f"{model} not found verified.", indent= INDENT_MULTIPLIER * 5)
            self.assertTrue(flag, f"File for model {model} does not exist")
            
        self.__class__.log("End verify_ml", indent= INDENT_MULTIPLIER * 4)

    def single_dataset_test(self, dataset, target, ignore_cols, categorical_cols, task):
        dir_name = 'Regression' if task == "MLRegression" else 'Classification'

        dataset_path = os.path.join("Datasets", dir_name, dataset)
        base_request_payload = {
            "name" : dataset,
            "task" : task,
            "dataset_path" : dataset_path,
            "target_column" : target,
            "ignore_cols" : ignore_cols,
            "path_to_save" : self.save_dir,
            "categorical_columns" : categorical_cols,
            "models" : None,
            "handle_imbalanced_data" : "" # removed in future but need for testing right now
        }
        for preprocessing in [True, False]:
            for imputation in [False, True]:
                for scaling in [False, True]:
                    for hyper_parameter_tuning in [0,1,2]:
                        for reduce_features in [False, True]:
                            final_payload = base_request_payload.copy()
                            final_payload.update(
                                {
                                    "preprocessing" : preprocessing,
                                    "imputation" : imputation,
                                    "scaling" : scaling,
                                    "hyper_parameter_tuning" : hyper_parameter_tuning,
                                    "reduce_features" : reduce_features
                                }
                            )
                            self.__class__.empty_directory(self.save_dir)
                            response = self.client.post("ml_predict", json = final_payload)
                            if response.status_code == 200:
                                self.__class__.log(f"200 Status code for {task} for dataset {dataset}. Payload: {final_payload}", indent= INDENT_MULTIPLIER * 4)
                            else:
                                self.__class__.log(f"Not 200 Status code for {task} for dataset {dataset}. Status Code: {response.status_code}. Payload: {final_payload}", indent= INDENT_MULTIPLIER * 4)
                            self.assertEqual(response.status_code, 200, f"Status code for {task} for dataset {dataset} was not 200. Status Code: {response.status_code}")
                            self.verify_ml(models=None, task=task, input_payload=final_payload)


    def test_regression(self):
        self.__class__.empty_directory(self.save_dir)
        self.__class__.log('-'*50 + "Start test_regression" + '-'*50, indent= INDENT_MULTIPLIER * 1)
        task = "MLRegression"
        regression_datesets = os.listdir(os.path.join("Datasets", "Regression"))
        regression_datesets = [dataset for dataset in regression_datesets if dataset.endswith(".csv")]
        targets_json = json.load(open(os.path.join("Datasets", "Regression", "targets.json"), "r"))
        ignore_cols_json = json.load(open(os.path.join("Datasets", "Regression", "ignore_cols.json"), "r"))
        categorical_cols_json = json.load(open(os.path.join("Datasets", "Regression", "categorical_cols.json"), "r"))
        self.__class__.log(f"Running test_regression for: {regression_datesets}", indent= INDENT_MULTIPLIER * 2)
        for dataset in regression_datesets:
            if not dataset.endswith('.csv'):
                continue
            self.__class__.log(f"Start test for {dataset}", indent= INDENT_MULTIPLIER * 3)
            self.single_dataset_test(dataset, targets_json[dataset], ignore_cols_json.get(dataset, []), categorical_cols_json.get(dataset, []), task=task)
            self.__class__.log(f"End test for {dataset}", indent= INDENT_MULTIPLIER * 3)
        self.__class__.log('-'*50 + "End test_regression" +'-'*50, indent= INDENT_MULTIPLIER * 2)
        
    
    def test_classification(self):
        self.__class__.empty_directory(self.save_dir)
        self.__class__.log('-'*50+ "Start test_classification" +'-'*50, indent= INDENT_MULTIPLIER * 1)

        task = "MLClassification"
        classification_dataset = os.listdir(os.path.join("Datasets", "Classification"))
        classification_dataset = [dataset for dataset in classification_dataset if dataset.endswith(".csv")]
        targets_json = json.load(open(os.path.join("Datasets", "Classification", "targets.json"), "r"))
        ignore_cols_json = json.load(open(os.path.join("Datasets", "Classification", "ignore_cols.json"), "r"))
        categorical_cols_json = json.load(open(os.path.join("Datasets", "Classification", "categorical_cols.json"), "r"))
        self.__class__.log(f"Running test_classification for: {classification_dataset}", indent= INDENT_MULTIPLIER * 2)
        for dataset in classification_dataset:
            if not dataset.endswith('.csv'):
                continue
            self.__class__.log(f"Start test for {dataset}", indent= INDENT_MULTIPLIER * 3)
            self.single_dataset_test(dataset, targets_json[dataset], ignore_cols_json.get(dataset, []), categorical_cols_json.get(dataset, []), task=task)
            self.__class__.log(f"End test for {dataset}", indent= INDENT_MULTIPLIER * 3)
        self.__class__.log('-'*50 + "End test_classification" + '-'*50, indent= INDENT_MULTIPLIER * 1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_dir)
        shutil.rmtree('Datasets')