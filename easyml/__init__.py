from easyml.core.controllers import *
try:
    from easyml.nlp.nlp_controllers import *
except ImportError:
    pass  # NLP deps not installed — skip (user can install with pip install easy-ml[nlp])
from easyml.utils.logger import write_log

class MLTrainer:
    """
    MLTrainer is a central orchestrator for managing machine learning tasks.

    It dynamically loads and invokes the appropriate controller based on the specified task
    (e.g., MLClassificationController, MLRegressionController).

    Attributes:
        controller: An instance of the selected controller class based on the task type.
    """

    def __init__(self, **args):
        """
        Initializes the MLTrainer by dynamically selecting the appropriate controller.

        Args:
            **args: A dictionary containing the task type and additional parameters required
                    to initialize the controller. It must include the key 'task'.

        Raises:
            ValueError: If the provided task is unsupported.
            Exception: Re-raises any exception that occurs during initialization after logging.
        """
        try:

            controller_class_name = f"{args['task']}Controller"
            write_log(f"entered MLTrainer __init__ for controller - {controller_class_name}", "info", os.path.basename(__file__))

            # Use globals() to fetch and instantiate the class
            if controller_class_name in globals():
                self.controller = globals()[controller_class_name](**args)  # Dynamically create the controller object
            else:
                raise ValueError(f"Unsupported task: {args['task']}")

        except Exception as e:
            if hasattr(self, 'controller') and hasattr(self.controller, 'request_body') and self.controller.request_body.status_code == 200:
                self.controller.request_body.error_message = "OtherError"
                self.controller.request_body.status_code = error_code_mappings[self.controller.request_body.error_message]
                self.controller.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.controller.request_body.error_message,
                    self.controller.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e

    def run(self):
        """
        Executes the controller's run method to process the ML pipeline.

        Returns:
            Any: The response returned by the controller's run method.

        Raises:
            Exception: Re-raises any exception that occurs during execution after logging.
        """
        try:
            response = self.controller.run()

            return response

        except Exception as e:
            if self.controller.request_body.status_code == 200:
                self.controller.request_body.error_message = "OtherError"
                self.controller.request_body.status_code = error_code_mappings[self.controller.request_body.error_message]
                self.controller.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.controller.request_body.error_message,
                    self.controller.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e