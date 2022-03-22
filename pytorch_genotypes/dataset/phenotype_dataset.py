"""
Implementation of a dataset that add phenotype data.
"""


from typing import Optional, List

import torch
import pandas as pd
import numpy as np

from .core import GeneticDataset, GeneticDatasetBackend


class PhenotypeGeneticDataset(GeneticDataset):
    """Dataset implementation for genotype-phenotype models.

    Uses an arbitrary backend for the genetic data and a pandas DataFrame to
    hold the correpsonding phenotypes.

    """
    def __init__(
        self,
        backend: GeneticDatasetBackend,
        phenotypes: pd.DataFrame,
        phenotypes_sample_id_column: str = "sample_id",
        exogenous_columns: Optional[List[str]] = None,
        endogenous_columns: Optional[List[str]] = None,
    ):
        super().__init__(backend)

        expected_cols = {phenotypes_sample_id_column}
        if exogenous_columns is not None:
            expected_cols &= set(exogenous_columns)

        if endogenous_columns is not None:
            expected_cols &= set(endogenous_columns)

        missing_cols = expected_cols - set(phenotypes.columns)
        if missing_cols:
            raise ValueError(f"Missing expected column(s): '{missing_cols}'.")

        self.overlap_samples(phenotypes[phenotypes_sample_id_column])

        # Infer dtype from backend.
        try:
            dtype = self.backend[0].dtype
            dtype = getattr(torch, getattr(dtype, "name", "float32"))
        except IndexError:
            dtype = torch.float32

        # Reorder phenotypes wrt the genetic dataset.
        phenotypes = phenotypes.iloc[self.idx["phen"], :]
        if exogenous_columns:
            self.exog: Optional[torch.Tensor] = torch.tensor(
                phenotypes.loc[:, exogenous_columns].values
            ).to(dtype)
        else:
            self.exog = None

        if endogenous_columns:
            self.endog: Optional[torch.Tensor] = torch.tensor(
                phenotypes.loc[:, endogenous_columns].values
            ).to(dtype)
        else:
            self.endog = None

    def __getitem__(self, idx):
        """Retrieve data at index.

        The return type is a tuple of length 1 to 3 depending on the requested
        endogenous and exogenous variables. The order is always:

            - (genotypes, exogenous, endogenous)

        """
        # Get the genotypes from the backend.
        geno = self.backend[self.idx["geno"][idx]]
        out = [geno]

        if self.exog is not None:
            out.append(self.exog[idx, :])

        if self.endog is not None:
            out.append(self.endog[idx, :])

        return tuple(out)

    def __len__(self):
        return len(self.idx["geno"])

    def overlap_samples(self, phenotype_samples):
        genetic_samples = self.backend.get_samples()
        overlap = set(genetic_samples) & set(phenotype_samples)

        genetic_samples_type = type(genetic_samples[0])
        phenotype_samples_type = type(phenotype_samples[0])
        if genetic_samples_type is not phenotype_samples_type:
            raise ValueError(
                f"Genetic file sample type: '{genetic_samples_type}' is "
                f"different from phenotype samples ('{phenotype_samples_type}'"
                ")."
            )

        if len(overlap) == 0:
            raise ValueError("No overlap between the genetic and phenotype "
                             "samples.")

        indexer = make_indexer(genetic_samples, phenotype_samples)
        self.idx = {
            "geno": indexer.left_idx.values,
            "phen": indexer.right_idx.values
        }


def make_indexer(left_samples, right_samples):
    left_df = pd.DataFrame({"left_id": left_samples})
    left_df["left_idx"] = np.arange(left_df.shape[0])

    right_df = pd.DataFrame({"right_id": right_samples})
    right_df["right_idx"] = np.arange(right_df.shape[0])

    df = pd.merge(left_df, right_df,
                  left_on="left_id", right_on="right_id",
                  how="outer")

    # The outer join forces floats for indices so that the missing values are
    # filled with NA. After dropna, we restore integer types for indices.

    df = df.dropna()

    for col in ("left_idx", "right_idx"):
        df[col] = df[col].astype(int)

    if df.shape[0] == 0:
        raise RuntimeError("Can't index non-overlapping datasets.")

    # Sort wrt left to minimize shuffling around on this dataset.
    df = df.sort_values("left_idx")

    return df
