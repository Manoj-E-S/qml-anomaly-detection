import pandas as pd

supported_readers = {
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".xlsx": pd.read_excel,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".h5": pd.read_hdf,
    ".html": pd.read_html,
}

unsupported_message = f"""
    Unsupported file extension '{{file_extension}}'.
    Supported extensions are: {', '.join(supported_readers.keys())}
"""

from .preprocessor import Preprocessor
