from typing import Optional, Set, Callable

import numpy as np
from geneparse.core import GenotypesReader, Genotypes


VariantPredicate = Callable[[Genotypes], bool]


def get_selected_samples_and_indexer(
    reader: GenotypesReader,
    keep_samples: Optional[Set[str]]
):
    """Utility function to overlap geneparse samples with selected IDs.

    This function returns the list of samples and their corresponding indices
    in the genotype file as a vector of ints suitable for indexing.

    """
    file_samples = reader.get_samples()

    if keep_samples is None:
        return file_samples, None

    file_samples_set = set(file_samples)
    overlap = file_samples_set & keep_samples

    genetic_sample_type = type(file_samples[0])
    keep_samples_type = type(next(iter(keep_samples)))

    if genetic_sample_type is not keep_samples_type:
        raise ValueError(
            f"Genetic file sample type: '{genetic_sample_type}' is "
            f"different from provided samples list ('{keep_samples_type}'"
            ")."
        )

    if len(overlap) == 0:
        raise ValueError(
            "No overlap between keep_samples and genetic dataset."
        )

    indices = []
    samples = []
    for index, sample in enumerate(file_samples):
        if sample in keep_samples:
            samples.append(sample)
            indices.append(index)

    return samples, np.array(indices, dtype=int)
