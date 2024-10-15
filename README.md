# Project Setup

Follow these steps to set up the `qml-anomaly-detection` project using Poetry:

1. **Install Poetry**:

    **Linux, macOS, Windows (WSL)**
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```
    **Windows Powershell:**
    ```sh
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```
    **Note:** On Windows you must add the pypoetry Scripts path to your `PATH` environment variable. Poetry by default is installed at `%APPDATA%\pypoetry` on Windows.

3. **Configure Poetry to use project directory for Virtual Environment**
    ```sh
    poetry config virtualenvs.in-project true
    ```

4. **Install required python version using `pyenv`**

    **Install pyenv:**
    ```sh
    sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    curl https://pyenv.run/ | bash
    ```
    **For Windows, refer to the installation here at https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#installation**
   
    **Install python 3.10:**
    ```sh
    pyenv install 3.10
    ```

6. **Clone the qml-anomaly-detection repository**
    ```sh
    git clone https://github.com/Manoj-E-S/qml-anomaly-detection.git
    ```

7. **Set python-version as the local(default) version for the project**
    ```sh
    cd qml-anomaly-detection
    pyenv local 3.10
    ```

8. **Install Dependencies under the python-version specified in pyproject.toml**:
    ```sh
    cd qml-anomaly-detection
    poetry env use 3.10
    poetry install
    ```

9. **Intall `torch` dependency**
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

9. **Download the datasets** 

    Download from [here](https://drive.google.com/file/d/1VDZccs-BXxPoLvGIkhFpVfTPiKd4LTWS/view?usp=sharing) and replace the contents in the `datasets` directory to form the following structure<br>
    /datasets/ <br>
    ├── ccfraud/ <br>
    ├── diabetes/ <br>
    ├── KDD Cup 1999/ <br>
    └── README.md <br>


10. **Run Your Project**:
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