import os
import random
from datetime import datetime
from enum import Enum

from . import log_folder_name


class __Logger:
    log_str = ""

    def __init__(self):
        pass

    def log(self, code: str, name: str, message: str) -> None:
        time_str = self.__get_time_string()
        self.log_str += f"\n[{time_str}] [{code}] [{name}] - {message}"

    def dump(self) -> None:
        log_file = self.__get_log_file_path()

        with open(log_file, "w") as log_file:
            log_file.write(self.log_str)

    def __get_log_file_path(self) -> str:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        random_int_str = str(random.randint(10000, 99999))
        file_name = f"CQU_LOGS_{current_time}_{random_int_str}.txt"

        return os.path.join(log_folder_name, file_name)

    def __get_time_string(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
