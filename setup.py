from setuptools import find_packages, setup

setup(
    name="cqu",
    version="0.1.1",
    author_email="johndoe@gmail.com",
    author="Manoj E S, Ian Sushruth Tauro, Kunal L, Aravind S N",
    description="CQU is a Classical and Quantum Machine Learning Library that is built on top of Qiskit and PyTorch, giving easy access to both classical and quantum machine learning algorithms through simple interfaces. It is designed to be easy to use and to be easily extensible, including a Preprocessor class that allows for easy integration of new data preprocessing techniques.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/Manoj-E-S/qml-anomaly-detection",
    packages=find_packages(),
    install_requires=[
        "qiskit",
        "torch",
    ],
)
