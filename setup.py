import os

from setuptools import find_packages, setup


# Package meta data
NAME = "inverted_index"
DESCRIPTION = "Student project to work with inverted index"
URL = "https://github.com/jambinoid/InvertedIndex"
AUTHOR = "Nikolay S. Lyudkevich"
EMAIL = "nikolai.lyudkevich@gmail.com"
REQUIRES_PYTHON = ">=3.0.0"

here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQIERED = f.read().split("\n")
except FileNotFoundError:
    REQIERED = []

about = {}
with open(os.path.join(here, "version.txt"), "r") as f:
    about["__version__"] = f.read()

if __name__ == "__main__":
    setup(
        name=NAME,
        version=about["__version__"],
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        packages=find_packages(exclude=["tests", "examples"]),
        install_requires=REQIERED,
        include_package_data=True,
        license="MIT License"
    )