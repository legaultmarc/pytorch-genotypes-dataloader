from importlib_metadata import entry_points
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch_genotypes",
    version="0.1",
    author="Marc-André Legault",
    author_email="legaultmarc@gmail.com",
    description="Utilities for data loading of genotypes as pytorch tensors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/legaultmarc/pytorch-genotypes-dataloader",
    project_urls={
        "Bug Tracker": (
            "https://github.com/legaultmarc/pytorch-genotypes-dataloader/issues"
        )
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    package_data={"pytorch_genotypes.tests": ["test_data/*"]},
    install_requires=[
        "cyvcf2",
        "numpy",
        "torch"
    ],
    entry_points={
        "console_scripts": [
            "pt-geno-block-trainer=pytorch_genotypes.block_trainer.cli:main"
        ]
    }
)
