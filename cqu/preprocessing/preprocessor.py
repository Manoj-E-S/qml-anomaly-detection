import os

import pandas as pd

from . import supported_readers, unsupported_message


class Preprocessor:
    file_path: str
    file_extension: str
    dataframe: pd.DataFrame

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.file_extension = None
        self.dataframe = None

        self.__validate_file_path()
        self.__read_dataset()

    def __validate_file_path(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        _, self.file_extension = os.path.splitext(self.file_path)
        self.file_extension = self.file_extension.lower()

        if self.file_extension not in supported_readers:
            raise ValueError(
                unsupported_message.format(file_extension=self.file_extension)
            )

    def __read_dataset(self) -> None:
        data = supported_readers[self.file_extension](self.file_path)

        if isinstance(data, list):
            self.dataframe = data[0]
        else:
            self.dataframe = data
