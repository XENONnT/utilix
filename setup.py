from setuptools import setup, find_packages

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setup(
    name="utilix",
    version="0.6.0",
    url='https://github.com/XENONnT/utilix',
    description="User-friendly interface to various utilities for XENON users",
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requires,
    python_requires=">=3.6",
    long_description=readme + '\n\n' + history,
)
