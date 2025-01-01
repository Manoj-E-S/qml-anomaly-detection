import os
import random
from datetime import datetime

from . import LOG_FOLDER_NAME


class __Logger:
    log_buffer: str
    module_name: str

    def __init__(self, module_name: str):
        self.log_buffer = ""
        self.module_name = module_name

    def log(self, name: str, message: str) -> None:
        time_str = self.__get_time_string()
        self.log_buffer += f"\n[{time_str}] [{name}] - {message}"

    def dump(self, header: str) -> None:
        if not os.path.exists(LOG_FOLDER_NAME):
            os.makedirs(LOG_FOLDER_NAME)

        current_time_title = datetime.now().strftime("%Y-%m-%d At %H:%M:%S")
        log_title = (
            f"[ CQU {self.module_name} LOGS ] [ Generated On {current_time_title} ]"
        )

        log_output = f"{log_title}\n{header}\n{self.log_buffer}"
        log_file_path = self.__get_log_file_path()

        with open(log_file_path, "w") as log_file:
            log_file.write(log_output)

        print(f"Log file generated at: {log_file_path}")

    def __get_log_file_path(self) -> str:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        random_int_str = str(random.randint(10000, 99999))
        file_name = f"{self.module_name}_{current_time}_{random_int_str}.txt"

        return os.path.join(LOG_FOLDER_NAME, file_name)

    def __get_time_string(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
