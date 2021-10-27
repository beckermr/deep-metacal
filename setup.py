import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "deep_metacal",
    "_version.py",
)
with open(pth, 'r') as fp:
    exec(fp.read())


setup(
    name='deep_metacal',
    description="code for metacal w/ wide- and deep-field data",
    author="Matthew R. Becker",
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    # use_scm_version=True,
    # setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
