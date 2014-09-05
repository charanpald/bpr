
import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext



ext_modules = [Extension("bprCython", ["bprCython.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3", ]),
]


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
    ], cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
