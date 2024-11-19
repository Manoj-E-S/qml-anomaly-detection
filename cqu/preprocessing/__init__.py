"""The preprocessing module providing all preprocessing tools for CQU.

This module provides the main 'Preprocessor' class that can be used to 
preprocess raw datasets into a more standardized and usable format. Use
this module to standardize your data before feeding it to the Quantum
Embedding Module or for model training.

Typical usage example:

>>> import cqu.preprocessing as cqupp
>>> pp = cqupp.Preprocessor("path/to/dataset")
>>> pp.get_missing_summary()
{ 'v1': 0, 'v2': 0, 'v3': 4, 'time': 6, 'class': 2 }

>>> strategies = { 
    'v3': cqupp.MissingValueStrategies.FILL_MEDIAN, 
    'time': cqupp.MissingValueStrategies.FILL_NOCB, 
    'class': cqupp.MissingValueStrategies.DROP_ROWS 
    }
>>> pp.clean_missing(strategies)
>>> pp.get_missing_summary()
{ 'v1': 0, 'v2': 0, 'v3': 0, 'time': 0, 'class': 0 }

"""

import pandas as pd

supported_readers = {
    ".csv": (pd.read_csv, "csv"),
    ".json": (pd.read_json, "json"),
    ".xlsx": (pd.read_excel, "excel"),
    ".parquet": (pd.read_parquet, "parquet"),
    ".feather": (pd.read_feather, "feather"),
    ".h5": (pd.read_hdf, "hdf"),
    ".html": (pd.read_html, "html"),
}

unsupported_message = f"""
    Unsupported file extension '{{file_extension}}'.
    Supported extensions are: {', '.join(supported_readers.keys())}
"""

from .missing_values import MissingValueStrategies
from .preprocessor import Preprocessor
