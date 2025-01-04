"""The logging module provides a internal Logger class to log cqu operations.

This module is not meant for direct use out of the box. It is used internally
by the other modules to log operations and errors. To log your own operations,
use the logging module from the Python standard library instead.
"""

LOG_FOLDER_NAME = "cqu_logs"

from .logger import Logger
