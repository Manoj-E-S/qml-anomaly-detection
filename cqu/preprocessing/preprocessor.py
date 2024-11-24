import os
import string
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, overload

import pandas as pd

from ..__logging import __Logger as Logger
from . import supported_readers, unsupported_message
from .missing_values import MissingValueStrategies, handle_missing_values
from .smote import smote_on_column
from .standardization import (
    StringStandardizers,
    filter_columns,
    standardize_numeric,
    standardize_strings,
)
from .type_conversion import convert_types


class Preprocessor:
    file_path: Optional[str]
    file_extension: Optional[str]
    dataframe: pd.DataFrame
    label_mappings: Dict[str, Dict[str, int]]
    log_text: str
    logger: Logger

    @overload
    def __init__(self, file_path: str) -> None: ...

    @overload
    def __init__(self, dataframe: pd.DataFrame) -> None: ...

    def __init__(
        self, dataset_input: str | pd.DataFrame, keep_duplicates: str = "first"
    ) -> None:
        self.file_path = None
        self.file_extension = None
        self.dataframe = None
        self.label_mappings = {}
        self.log_text = ""
        self.logger = Logger("PREPROCESSOR")

        if keep_duplicates not in {"first", "last", False}:
            raise ValueError(
                "Invalid value for keep_duplicates. Please provide 'first', 'last', or False."
            )

        if isinstance(dataset_input, str):
            self.file_path = dataset_input
            self.__validate_file_path()
            self.__read_dataset()
        elif isinstance(dataset_input, pd.DataFrame):
            self.dataframe = dataset_input
            self.logger.log("INIT", "Dataset loaded successfully from DataFrame")
        else:
            raise ValueError(
                "Invalid input type. Please provide a file path or a DataFrame."
            )

        self.__handle_columns()
        self.dataframe = self.dataframe.infer_objects()
        self.logger.log(
            "INIT", "Data types inferred using pd.DataFrame.infer_objects()"
        )
        self.__handle_duplicate_rows(keep_duplicates)

    def get_missing_summary(self) -> Dict[str, int]:
        return self.dataframe.isnull().sum().to_dict()

    @overload
    def clean_missing(
        self, strategy: MissingValueStrategies = MissingValueStrategies.DROP_COLUMNS
    ) -> None: ...

    @overload
    def clean_missing(self, strategies: Dict[str, MissingValueStrategies]) -> None: ...

    def clean_missing(
        self,
        strategy_or_strategies: (
            MissingValueStrategies | Dict[str, MissingValueStrategies]
        ) = MissingValueStrategies.DROP_COLUMNS,
    ) -> None:
        self.dataframe = handle_missing_values(self.dataframe, strategy_or_strategies)

        if isinstance(strategy_or_strategies, dict):
            for column, strategy in strategy_or_strategies.items():
                self.logger.log(
                    "CLEAN_MISSING",
                    f"Missing values in column '{column}' handled using strategy: {strategy}",
                )
        else:
            self.logger.log(
                "CLEAN_MISSING",
                f"Missing values in entire DataFrame handled using strategy: {strategy_or_strategies}",
            )

    def convert_dtypes(self, column_types: Dict[str, Any]) -> None:
        self.dataframe = convert_types(self.dataframe, column_types)

        for column, dtype in column_types.items():
            self.logger.log(
                "DTYPE_CONVERSION", f"Column '{column}' converted to type: {dtype}"
            )

    def standardize_numeric_data(self, columns: Optional[List[str]] = None) -> None:
        self.dataframe = standardize_numeric(self.dataframe, columns)

        if columns:
            for column in columns:
                self.logger.log(
                    "STANDARDIZATION_NUMERIC",
                    f"Column '{column}' standardized using Z-score normalization",
                )
        else:
            self.logger.log(
                "STANDARDIZATION_NUMERIC",
                "All numeric columns standardized using Z-score normalization",
            )

    @overload
    def standardize_string_data(
        self, standardizer: StringStandardizers = StringStandardizers.CLEAN
    ) -> None: ...

    @overload
    def standardize_string_data(
        self, standardizers: Dict[str, StringStandardizers]
    ) -> None: ...

    def standardize_string_data(
        self,
        standardizer: (
            StringStandardizers | Dict[str, StringStandardizers]
        ) = StringStandardizers.CLEAN,
    ) -> None:
        self.dataframe, mappings = standardize_strings(self.dataframe, standardizer)
        self.label_mappings.update(mappings)

        if isinstance(standardizer, dict):
            for column, strategy in standardizer.items():
                self.logger.log(
                    "STANDARDIZATION_STRING",
                    f"Column '{column}' standardized using strategy: {strategy}",
                )
        else:
            self.logger.log(
                "STANDARDIZATION_STRING",
                f"All string columns standardized using strategy: {standardizer}",
            )

    def filter_columns(self, columns: Dict[str, Callable[[Any], bool]]) -> None:
        row_count = self.dataframe.shape[0]

        self.dataframe = filter_columns(self.dataframe, columns)

        rows_filtered_out = row_count - self.dataframe.shape[0]

        for column, filter_func in columns.items():
            self.logger.log(
                "FILTERING",
                f"Rows in column '{column}' filtered using custom function. {rows_filtered_out} rows filtered out.",
            )

    def smote_on_column(
        self, target_column: str, random_state: int = None, k_neighbors: int = 5
    ) -> None:
        self.dataframe = smote_on_column(
            self.dataframe, target_column, random_state, k_neighbors
        )
        self.logger.log(
            "SMOTE",
            f"Oversampling performed with column '{target_column}' as class using SMOTE",
        )

    def generate_logfile(self) -> None:
        log_header = f"""
[DATASET SHAPE]: {self.dataframe.shape}\n
[COLUMNS]: {', '.join(self.dataframe.columns)}\n
[MISSING VALUES]: {self.get_missing_summary()}\n
[LABEL MAPPINGS]: {self.label_mappings}
        """
        self.logger.dump(log_header)

    def write_to(self, file_path: str) -> None:
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension not in supported_readers:
            raise ValueError(unsupported_message.format(file_extension=extension))

        method_name = f"to_{supported_readers[extension][1]}"
        method = getattr(self.dataframe, method_name)
        method(file_path)

        self.logger.log("FILE_WRITE", f"DataFrame written to file: {file_path}")

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
        data = supported_readers[self.file_extension][0](self.file_path)

        if isinstance(data, list):
            self.dataframe = data[0]
        else:
            self.dataframe = data

        self.logger.log(
            "INIT", f"Dataset loaded successfully from path: {self.file_path}"
        )

    def __handle_columns(self) -> None:
        self.dataframe.columns = (
            self.dataframe.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
        )
        self.dataframe = self.dataframe.loc[:, ~self.dataframe.columns.duplicated()]
        self.logger.log(
            "INIT",
            "Column names standardized to all lowercase and underscores seperated strings",
        )

    def __handle_duplicate_rows(self, keep: str) -> None:
        self.dataframe = self.dataframe.drop_duplicates(keep=keep).reset_index(
            drop=True
        )
        self.logger.log("INIT", "Duplicate rows deleted from DataFrame")
