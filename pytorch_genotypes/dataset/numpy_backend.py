"""
Numpy backend for genotypes for datasets that fit in memory.
"""

import os
from typing import Optional, Set, Iterable, List

from tqdm import tqdm
import torch
import numpy as np
from numpy.typing import DTypeLike
from geneparse.core import GenotypesReader, Variant

from .core import GeneticDatasetBackend
from .utils import VariantPredicate, get_selected_samples_and_indexer


class NumpyBackend(GeneticDatasetBackend):
    def __init__(
        self,
        reader: GenotypesReader,
        npz_filename: str,
        keep_samples: Optional[Set[str]] = None,
        variant_predicates: Optional[Iterable[VariantPredicate]] = None,
        dtype: DTypeLike = np.float16,
        progress: bool = True
    ):
        self.samples, self._idx = get_selected_samples_and_indexer(
            reader, keep_samples
        )
        self.npz_filename = os.path.abspath(npz_filename)
        self.variants: List[Variant] = []

        self.create_np_matrix(reader, variant_predicates, dtype, progress)

    def create_np_matrix(
        self,
        reader: GenotypesReader,
        variant_predicates: Optional[Iterable[VariantPredicate]],
        dtype: DTypeLike,
        progress: bool
    ):
        n_variants = reader.get_number_variants()

        m = np.empty((self.get_n_samples(), n_variants), dtype=dtype)
        cur_column = 0
        variants = []
        if progress:
            iterator = tqdm(reader.iter_genotypes(), total=n_variants)
        else:
            iterator = reader.iter_genotypes()

        for g in iterator:
            if any([not f(g) for f in variant_predicates or []]):
                continue

            if self._idx is not None:
                genotypes = g.genotypes[self._idx]
            else:
                genotypes = g.genotypes

            m[:, cur_column] = genotypes.astype(dtype)
            variants.append(g.variant)
            cur_column += 1

        # Resize if some variants were filtered out.
        m = m[:, :cur_column]

        self.variants = variants
        self.m = m
        np.savez_compressed(self.npz_filename, m)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("m")
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self.m = np.load(self.npz_filename)["arr_0"]

    def __getitem__(self, idx):
        return torch.tensor(self.m[idx, :])

    def get_samples(self):
        return self.samples

    def get_variants(self):
        return self.variants

    def get_n_samples(self):
        return len(self.samples)

    def get_n_variants(self):
        return len(self.variants)
