# entry point
from src.core.pre_processing import *
from src.core.model_builder import *
from src.utils.data_loader import *
from src.config.GLOBAL_VARS import *
from src.core.controllers import *


class MLTrainer:
    def __init__(self, **args):

        controller_class_name = f"{args['task']}Controller"
        print(controller_class_name)

        # print(dataset_path, target_column, preprocessing, imputation, scaling,
        #          models, hyper_parameter_tuning, verbose, categorical_variables, path_to_save,
        #          reduce_features, handle_imbalanced_data, task, sep="/n")

        print(args)

        # Use globals() to fetch and instantiate the class
        if controller_class_name in globals():
            self.controller = globals()[controller_class_name](**args)  # Dynamically create the controller object
        else:
            raise ValueError(f"Unsupported task: {args['task']}")

    def run(self):
        response = self.controller.run()

        return response

