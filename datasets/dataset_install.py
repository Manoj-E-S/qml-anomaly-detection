import gdown
import os

# Public Dataset GDrive File ID
GD_DATASET_FILE_ID = '1VDZccs-BXxPoLvGIkhFpVfTPiKd4LTWS'
DATASET_FOLDER = 'datasets'
DATASET_ZIP_OUTPUT = f'{DATASET_FOLDER}{os.sep}datasets.zip'

print("\nPlease wait while the dataset is being installed...\n")

# Download
gdown.download(f'https://drive.google.com/uc?id={GD_DATASET_FILE_ID}', DATASET_ZIP_OUTPUT, quiet=False)

print("\nDataset has been installed successfully!\n")