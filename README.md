# Introduction

This project aims to facilitate working with genotype data in Pytorch.

In addition to providing read access to VCF files, this package allows chunking the file into a fixed number of genetic variants by constructing an ``sqlite`` database of variant metadata.

We also provide tools to create Pytorch datasets from genotype and/or phenotype
data.

# Installation

```bash
git clone git@github.com:legaultmarc/pytorch-genotypes-dataloader.git
pip install ./pytorch-genotypes-dataloader
```

# Example

## Creating a Pytorch dataset of phenotype and genotype data.

First, we need to initialize a backend for access to the genotypes. Backends
are meant to abstract away storage and data sources while providing reasonably
fast _per sample_ data access (as opposed to most genotype file format which 
provide _per variant_ data access).

For now, the only implementation is using [Zarr](https://zarr.readthedocs.io/en/stable/)
to recode all the genotypes.

This backend can take any [geneparse](https://github.com/pgxcentre/geneparse)
reader as input allowing support for most commonly used variant formats (_e.g._
plink, bgen, etc.)

```python
# Creating the zarr backend from a geneparse reader.
import geneparse
from pytorch_genotypes.dataset import ZarrBackend, PhenotypeGeneticDataset

reader = geneparse.parsers["bgen"](
    "my_filename.bgen", sample_filename="my_filename.sample"
)

backend = ZarrBackend(
    reader,  # Any geneparse reader
    filename_prefix: "output_zarr_filename",  # Output filename
)
# Note that there are many other options for variant and sample selection.

# Load phenotype data into a Pandas DataFrame.
phenotypes = pd.read_csv("my_phenotype_filename.csv")

dataset = PhenotypeGeneticDataset(
    backend,
    phenotypes,
    exogenous_columns=...,  # Add covariable names here
    endogenous_columns=...,  # Add outcome variable names here
)

dataset[0]
# Tuple of tensors corresponding to the genotypes, exogenous variables and
# endogenous variables.

# dataset is a Pytorch dataset.

```

## Creating chunks from a VCF file.

```python
import pytorch_genotypes
vcf = FixedSizeVCFChunks("my_vcf_file.vcf.gz", create=True)
# Example output:
# Using DB: /home/legaultm/projects/pytorch-genotypes-dataloader/all_1kg_chr1_phased_GRCh38_snps_maf0.01.recode.db
# <Chunk #0 - 1:16103-47128618>
# <Chunk #1 - 1:47128702-97377422>
# <Chunk #2 - 1:97377695-171680796>
# <Chunk #3 - 1:171680851-220672927>
# <Chunk #4 - 1:220672933-248945660>

# Create the pytorch tensor for the region corresponding to chunk #1.
m = vcf.get_tensor_for_chunk_id(1)
m.shape
# torch.Size([2548, 200000])
```

Note that the `.db` file is a regular sqlite database which holds the chunk
and variant metadata. You can use sqlite3 to see the schema and contents after
creation. Once it has been created using `create=True`, there is **no need** to create it again.

By default, the chunks hold 200k variants, but this can be changed easily.
