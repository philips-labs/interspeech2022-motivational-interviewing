import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="interspeechmi-asanohideo549",
    version="0.0.1",
    author="Zixiu (Alex) Wu",
    description="Python code for Next Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires='>=3.9.7',
)
