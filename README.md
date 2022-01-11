# Introduction

This project aims to facilitate access to genotypes stored in [VCF](http://samtools.github.io/hts-specs/VCFv4.3.pdf) files as [numpy](https://numpy.org/doc/stable/index.html) arrays or [pytorch](https://pytorch.org/) tensors.

In addition to providing read access to VCF files, this package allows chunking the file into a fixed number of genetic variants by constructing an ``sqlite`` database of variant metadata.

# Installation

```bash
git clone git@github.com:legaultmarc/pytorch-genotypes-dataloader.git
pip install ./pytorch-genotypes-dataloader
```

# Example

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

# Features and roadmap

This project is very young and most non-trivial tasks require querying
the sqlite3 database manually. This is relatively straightforward and the VCF object holds a reference to a database connection object (`.con`) which can be used for this purpose.

In the future, I aim to make it more easy to search and extract specific variants or genomic regions.

I would also like to support other file formats.

Feature requests and contributions are welcome.