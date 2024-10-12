# Project Setup

Follow these steps to set up the `qml-anomaly-detection` project using Poetry:

1. **Install Poetry**:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Configure Poetry to use project directory for Virtual Environment**
    ```sh
    poetry config virtualenvs.in-project true
    ```

3. **Clone the qml-anomaly-detection repository**
    ```sh
    git clone https://github.com/Manoj-E-S/qml-anomaly-detection.git
    ```

4. **Install Dependencies under the python-version specified in pyproject.toml**:
    ```sh
    cd qml-anomaly-detection
    poetry env use 3.10
    poetry install
    ```

5. **Run Your Project**:
    ```sh
    poetry shell
    python <your_script>.py
    ```

We are good to go!

### Some useful Poetry Commands

1. **Create a New Poetry Project**:
    ```sh
    poetry new <project-name>
    cd <project-name>
    ```

2. **Add Dependencies**:
    ```sh
    poetry add <package-1-name> <package-2-name> ...
    ```

3. **Add Group Dependencies (dev-dependencies, test-dependencies, etc)**:
    ```sh
    poetry add --group <group-name> <package-1-name> <package-2-name> ...
    ```

4. **Add New Sources**:
    ```sh
    poetry source add --priority=<default|primary|supplemental|explicit> <custom-source-name> <source-url>
    ```

5. **Bring poetry.lock up-to-date with pyproject.toml (without updating dependencies)**:
    ```sh
    poetry lock --no-update
    ```

6. **Activate the Virtual Environment**:
    ```sh
    poetry shell
    ```

7. **Deactivate the Virtual Environment**:
    ```sh
    exit
    ```

For more details, refer to the [Poetry documentation](https://python-poetry.org/docs/).