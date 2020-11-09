from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import find_packages, setup, Command


def version():
    with open("autocti/__init__.py") as f:
        lines = f.read().split("\n")
    for line in lines:
        if "__version__" in line:
            return line.split("=")[1].strip(" '\”")


class RunTests(Command):
    """Run all tests."""

    description = "run tests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(["py.test", "--cov=autocti", "--cov-report=term-missing"])
        raise SystemExit(errno)

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

setup(
    name="autocti",
    version=version(),
    description="Automated Charge Transfer Inefficiency Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jammy2211/PyAutoCTI",
    author="James Nightingale and Richard Hayes",
    author_email="james.w.nightingale@durham.ac.uk",
    include_package_data=True,
    license="MIT License",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="cli",
    install_requires=requirements,
    extras_require={"test": ["coverage", "pytest", "pytest-cov"]},
    cmdclass={"test": RunTests},
)
