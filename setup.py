import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparselearning", # Replace with your own username
    version="0.1",
    author="Anonymous hippo, Anonymous potato",
    author_email="hippo@lemon.edu, potato@lemon.com",
    description="Rethinking Dynamic Sparse learning techniques.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hippo/rethinking-sparse-learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)