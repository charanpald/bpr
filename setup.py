
import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = "bpr",
    version = "0.1",
    author = "Mark Levy",
    author_email = "??",
    description = ("Implementation of BPR"),
    license = "??",
    keywords = "",
    url = "https://github.com/gamboviol/bpr",
    packages=[],
    install_requires=[],
    long_description="Implementation of BPR",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities"
    ],
)
