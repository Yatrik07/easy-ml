# entry point
from src.core.pre_processing import *
from src.core.model_builder import *
from src.utils.data_loader import *
from src.config.GLOBAL_VARS import *
from src.core.controllers import *
from src.utils.logger import write_log

class MLTrainer:
    def __init__(self, **args):

        try:

            controller_class_name = f"{args['task']}Controller"
            write_log(f"entered MLTrainer __init__ for controller - {controller_class_name}", "info", os.path.basename(__file__))

            # Use globals() to fetch and instantiate the class
            if controller_class_name in globals():
                self.controller = globals()[controller_class_name](**args)  # Dynamically create the controller object
            else:
                raise ValueError(f"Unsupported task: {args['task']}")

        except Exception as e:
            if self.controller.request_body.status_code == 200:
                self.controller.request_body.status_code = ERROR_CODE
                self.controller.request_body.error_message = str(e)
                self.controller.request_body.error_traceback = traceback.format_exc()

            raise e

    def run(self):

        try:
            response = self.controller.run()

            return response

        except Exception as e:
            if self.controller.request_body.status_code == 200:
                self.controller.request_body.status_code = ERROR_CODE
                self.controller.request_body.error_message = str(e)
                self.controller.request_body.error_traceback = traceback.format_exc()

                write_log("error occurred- {} - {}".format(
                    self.controller.request_body.error_message,
                    self.controller.request_body.error_traceback),
                    "error", os.path.basename(__file__))

            raise e