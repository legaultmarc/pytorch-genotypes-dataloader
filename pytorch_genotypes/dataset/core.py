"""
Abstract classes for datasets and dataset backends.

The idea is that the backends provide access to genetic data wheareas datasets
can implement additional logic, for example to include phenotype data.

"""

import pickle
from collections import defaultdict
from typing import List, Type, TypeVar, Tuple

import torch
from geneparse import Variant

from torch.utils.data.dataset import Dataset


T = TypeVar("T", bound="GeneticDatasetBackend")


class GeneticDatasetBackend(object):
    def get_samples(self) -> List[str]:
        raise NotImplementedError()

    def get_variants(self) -> List[Variant]:
        raise NotImplementedError()

    def get_n_samples(self) -> int:
        raise NotImplementedError()

    def get_n_variants(self) -> int:
        raise NotImplementedError()

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @classmethod
    def load(cls: Type[T], filename: str) -> T:
        with open(filename, "rb") as f:
            o = pickle.load(f)

        assert isinstance(o, cls)
        return o

    def __getitem__(self, idx) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.get_n_samples()


class GeneticDataset(Dataset):
    def __init__(self, backend: GeneticDatasetBackend):
        super().__init__()
        self.backend = backend

    def load_full_dataset(self) -> Tuple[torch.Tensor, ...]:
        """Utility function to load everything in memory.

        This is useful for testing datasets or to use with models that don't
        train by minibatch.

        """
        tensors = defaultdict(list)

        for i in range(len(self)):
            datum: Tuple[torch.Tensor, ...] = self[i]

            for j in range(len(datum)):
                tensors[j].append(datum[j])

        # Merge everything.
        return tuple((
            torch.vstack(tensors[j]) for j in range(len(tensors))
        ))

    # Autmatically dispatch to backend by default.
    def __getattr__(self, attr):
        if hasattr(self.backend, attr):
            return getattr(self.backend, attr)

        return super().__getattr__(attr)

    def __len__(self) -> int:
        return len(self.backend)
