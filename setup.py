import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparselearning", # Replace with your own username
    version="0.1",
    author="Varun Sundar, Rajat Vadiraj",
    author_email="vsundar4@wisc.edu, rajatvd@gmail.com",
    description="RigL (Evci et al. 2020) reproducibility.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/varun19299/rigl-reproducibility",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)