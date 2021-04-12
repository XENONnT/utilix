from setuptools import setup, find_packages

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

with open('README.md') as file:
    readme = file.read()

setup(
    name="utilix",
    version="0.5.0",
    url='https://github.com/XENONnT/utilix',
    description="User-friendly interface to various utilities for XENON users",
    #long_description=readme,
    packages=find_packages(),
    install_requires=requires,
    python_requires=">=3.6",
)
