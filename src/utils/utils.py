class EasyAIException(Exception):
    def __init__(self, message, error_code="E000", details=None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self):
        return {
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details
        }

#
# class PreprocessingError(EasyAIException):
#     def __init__(self, message="Preprocessing failed", details=None):
#         super().__init__(message, error_code="500", details=details)
#
#
# class ModelTrainingError(EasyAIException):
#     def __init__(self, message="Model training failed", details=None):
#         super().__init__(message, error_code="520", details=details)