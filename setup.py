import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="azhou-taoni", # Replace with your own username
    version="0.0.1",
    author="Alicia Zhou, Tao Ni",
    author_email="alicia.zhou@duke.edu, tao.ni@duke.edu",
    description="SGHMC package",
    url="https://github.com/azhou96/STA663-Final-Project-SGHMC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)