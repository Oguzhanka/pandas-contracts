from setuptools import setup, find_packages

setup(
    name="pandas-contracts",
    version="0.1.0",
    description="A helper library for validating and coercing pandas objects.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Oguzhan Karaahmetoglu",
    author_email="",
    url="https://github.com/Oguzhanka/pandas-contracts",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas==2.2.3",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
    },
)
