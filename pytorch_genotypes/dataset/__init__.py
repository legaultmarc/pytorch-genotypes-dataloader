# flake8: noqa
from .core import GeneticDataset, GeneticDatasetBackend
from .zarr_backend import ZarrBackend
from .numpy_backend import NumpyBackend
from .phenotype_dataset import PhenotypeGeneticDataset

BACKENDS = {
    "ZarrBackend": ZarrBackend,
    "NumpyBackend": NumpyBackend
}
