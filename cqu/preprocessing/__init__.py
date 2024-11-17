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

from .preprocessor import Preprocessor
