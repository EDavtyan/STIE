from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stie",
    version=1.0,
    author="Edgar Davtyan",
    author_email="edgar.davtyan@picsart.com",
    description="Sequential testing for early stopping of interleaving experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
        "tqdm",
        "scipy",
        "matplotlib",
        "s3fs"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)