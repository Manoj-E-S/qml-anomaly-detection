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

3. **Install required python version using `pyenv`**
    ```sh
    pyenv install python3.10
    ```

4. **Clone the qml-anomaly-detection repository**
    ```sh
    git clone https://github.com/Manoj-E-S/qml-anomaly-detection.git
    ```

5. **Set python-version as the local(default) version for the project**
    ```sh
    cd qml-anomaly-detection
    pyenv local 3.10
    ```

6. **Install Dependencies under the python-version specified in pyproject.toml**:
    ```sh
    cd qml-anomaly-detection
    poetry env use 3.10
    poetry install
    ```

7. **Intall `torch` dependency**
    1. If cuda is available:
        - Check cuda version on your device

            ```sh
            nvcc --version
            ```
        - Add corresponding `torch` repository source in `pyproject.toml` (Although this will not be useful in installing pytorch (as it will be installed outside of the dependency management), doing this will be helpful for documentation purposes)

            ```
            [[tool.poetry.source]]
            name = "pytorch-gpu-<PLACEHOLDER>"
            url = "https://download.pytorch.org/whl/<PLACEHOLDER>"
            priority = "explicit"
            ```
        - Install `torch` in the venv, without updating pyproject.toml

            ```sh
            cd qml-anomaly-detection
            poetry run pip install "torch~=2.4.1+<PLACEHOLDER>" --index-url "https://download.pytorch.org/whl/<PLACEHOLDER>"
            ```
        - *Note:* Replace **PLACEHOLDER** with cuda-appropriate repository. Ex: cu121 (for cuda 12), cu111 (for cuda 11), etc.
    
    2. If cuda is not available:

        ```sh
        cd qml-anomaly-detection
        poetry run pip install "torch~=2.4.1+cpu" --index-url "https://download.pytorch.org/whl/cpu/torch_stable.html"
        ```

8. **Run Your Project**:
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
    poetry add "<package-name>@<version-specifier>" --source <custom-source-name>
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