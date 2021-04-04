# example usage
# pip install git+https://github.com/deeptendies/deeptendies#egg=deeptendies

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeptendies",  # Replace with your own username
    version="0.0.1.dev",
    author="stancsz, mklasby, hasnil",
    author_email="deeptendies@deeptendies.github.io",
    description="deeptendies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deeptendies/deeptendies",
    project_urls={
        "Bug Tracker": "https://github.com/deeptendies/deeptendies/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)