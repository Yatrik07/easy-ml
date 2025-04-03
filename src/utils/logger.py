from datetime import datetime
import os
import logging

#TODO : this the final logger class, now just need to apply throughout our code to keep the track
class Logging:
    def __init__(self, filename, logger_filename):
        self.filename = filename or os.path.basename(__file__)
        self.file = open(logger_filename, "a")

    def create_log(self, message: str, logging_level: str) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_level = logging_level.upper()

        # Highlight WARNINGS and ERRORS
        if log_level == "WARNING":
            log_level = f"⚠️  {log_level}"
        elif log_level == "ERROR":
            log_level = f"❌ {log_level}"

        return f"{timestamp} | {log_level} | {self.filename} | {message}\n"

    def log(self, message):
        log = self.create_log(message, "info")
        self.file.write(log)
        self.file.flush()

    def warning(self, message):
        log = self.create_log(message, "warning")
        self.file.write(log)
        self.file.flush()

    def error(self, message):
        log = self.create_log(message, "error")
        self.file.write(log)
        self.file.flush()

    def close(self):
        self.file.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass

class InfoLevelFilter(logging.Filter):
    """
    Filter logs based on minimum log level
    """
    def __init__(self):
        super().__init__()
    
    def filter(self, record):
        return record.levelno == 20

logger = logging.getLogger('custom_logger')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs.log")
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
file_handler.addFilter(InfoLevelFilter())
file_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler("error_logs.log")
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
error_handler.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.addHandler(error_handler)

def get_logger():
    return logger