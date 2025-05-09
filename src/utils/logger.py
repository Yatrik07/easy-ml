from datetime import datetime
import os


#TODO : this the final logger class, now just need to apply throughout our code to keep the track
class logging:
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


logger_file_name = r"D:\PycharmProjects\easy-ml\sample_logs\application_logs.log"


def write_log(message, level, module_name):
    with open(logger_file_name, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = level.upper()
        f.write(f"{timestamp} - {module_name} - {level} - {message}\n")
