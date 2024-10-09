# Project Setup

Follow these steps to set up the `qml-tryout` project using Poetry:

1. **Install Poetry**:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Configure Poetry to use project directory for Virtual Environment**
    ```sh
    poetry config virtualenvs.in-project true
    ```

3. **Create a New Poetry Project**:
    ```sh
    poetry new qml-tryout
    cd qml-tryout
    ```

4. **Add Dependencies**:
    ```sh
    poetry add <package-1-name> <package-2-name> ...
    ```

5. **Add Group Dependencies (dev-dependencies, test-dependencies, etc)**:
    ```sh
    poetry add --group <group-name> <package-1-name> <package-2-name> ...
    ```

6. **Add New Sources**:
    ```sh
    poetry source add --priority=<default|primary|supplemental|explicit> <custom-source-name> <source-url>
    ```

7. **Bring poetry.lock up-to-date with pyproject.toml (without updating dependencies)**:
    ```sh
    poetry lock --no-update
    ```

7. **Install Dependencies under the python-version specified in pyproject.toml**:
    ```sh
    poetry env use <python-version>
    poetry install
    ```

8. **Activate the Virtual Environment**:
    ```sh
    poetry shell
    ```

9. **Deactivate the Virtual Environment**:
    ```sh
    exit
    ```

9. **Run Your Project**:
    ```sh
    poetry shell
    python <your_script>.py
    ```

For more details, refer to the [Poetry documentation](https://python-poetry.org/docs/).