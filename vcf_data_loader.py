from dataclasses import dataclass
import typing
import sqlite3
import gzip
import os

import cyvcf2
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


VERBOSE = True


def set_verbose(b: bool):
    global VERBOSE
    VERBOSE = b


@dataclass
class VCFChunk:
    chunk_id: int
    chrom: str
    start: int
    end: int

    @classmethod
    def new(cls, id, first_variant):
        return cls(id, first_variant.CHROM, first_variant.POS, None)

    def next_new(self, first_variant):
        return self.new(self.chunk_id + 1, first_variant)

    def sql_insert(self, cur):
        cur.execute(
            "insert into chunks values (?, ?, ?, ?)",
            (self.chunk_id, self.chrom, self.start, self.end)
        )

    def __repr__(self):
        return (
            f"<Chunk #{self.chunk_id} - {self.chrom}:{self.start}-{self.end}>"
        )

class FixedSizeVCFChunks(object):
    """Create chunks of variants from a VCF."""

    def __init__(self, vcf_filename, max_snps_per_chunk=200000, create=False):
        self.vcf_filename = vcf_filename

        db_filename = os.path.abspath(self._get_db_name())
        print(f"Using DB: {db_filename}")
        self.con = sqlite3.connect(db_filename)
        if create:
            self.create(max_snps_per_chunk)

    def _get_db_name(self):
        if ".vcf.gz" in self.vcf_filename:
            return self.vcf_filename.replace(".vcf.gz", ".db")

        elif ".vcf" in self.vcf_filename:
            return self.vcf_filename.replace(".vcf", ".db")

        else:
            return "vcf_chunks.db"

    def _sql_create(self):
        cur = self.con.cursor()

        cur.execute("drop table if exists chunks")
        cur.execute(
            "create table chunks ( "
            "  id integer primary key, "
            "  chrom text not null, "
            "  start integer not null, "
            "  end integer not null"
            ");"
        )

        cur.execute("drop table if exists variants")
        cur.execute(
            "create table variants ( "
            "  chunk_id integer, "
            "  chrom text not null, "
            "  pos integer not null, "
            "  ref text not null, "
            "  alt text not null, "
            "  constraint chunk_fk foreign key (chunk_id) references chunks (id)"
            ");"
        )
        self.con.commit()

    def create(self, max_snps_per_chunk):
        self._sql_create()
        cur_chunk = None
        cur_n = 0
        buf = []
        vcf_iter = iter_vcf_wrapper(cyvcf2.VCF(self.vcf_filename, lazy=True))
        prev = None
        for v in vcf_iter:
            if cur_chunk is None:
                # Initialize first chunk.
                cur_chunk = VCFChunk.new(id=0, first_variant=v)

            if cur_chunk.chrom != v.CHROM or cur_n >= max_snps_per_chunk:
                self._close_chunk(cur_chunk, last_variant=prev)
                cur_chunk = cur_chunk.next_new(first_variant=v)
                cur_n = 0

            buf.append([cur_chunk.chunk_id, v.CHROM, v.POS, v.REF, v.ALT[0]])
            cur_n += 1

            if len(buf) >= 1e6:
                buf = self._flush_buffer(buf)

            prev = v

        if buf:
            self._flush_buffer(buf)

        self._close_chunk(cur_chunk, last_variant=v)

    def _flush_buffer(self, buf):
        cur = self.con.cursor()
        cur.executemany("insert into variants values (?, ?, ?, ?, ?)", buf)
        return []

    def _close_chunk(self, cur_chunk: VCFChunk, last_variant):
        # Increment chunk counter and add entry to the db.
        cur_chunk.end = last_variant.POS
        cur = self.con.cursor()
        cur_chunk.sql_insert(cur)
        if VERBOSE:
            print(cur_chunk)
        self.con.commit()

    def iter_vcf_by_chunk_id(
        self,
        chunk_id: int
    ) -> typing.Generator[cyvcf2.Variant, None, None]:
        cur = self.con.cursor()
        cur.execute(
            "select chrom, start, end from chunks where id=?", (chunk_id, )
        )
        chrom, start, end = cur.fetchone()
        return iter_vcf_wrapper(
            cyvcf2.VCF(self.vcf_filename)(f"{chrom}:{start}-{end}")
        )

    def get_n_chunks(self):
        cur = self.con.cursor()
        cur.execute("select count(*) from chunks")
        return cur.fetchone()[0]

    def get_tensor_for_chunk_id(self, chunk_id):
        # Check how many samples and variants to pre-allocate memory.
        try:
            vcf = cyvcf2.VCF(self.vcf_filename)
            n_samples = len(vcf.samples)
        finally:
            vcf.close()

        cur = self.con.cursor()
        cur.execute(
            "select count(*) from variants where chunk_id=?", (chunk_id, )
        )
        n_snps = cur.fetchone()[0]

        mat = np.empty((n_samples, n_snps), dtype=np.float32)

        for j, v in enumerate(self.iter_vcf_by_chunk_id(chunk_id)):
            mat[:, j] = parse_vcf_genotypes(v.genotypes, format="additive")

        return torch.from_numpy(mat)


def iter_vcf_wrapper(vcf, biallelic=True, snp=True):
    """Wrapper over cyvcf2.VCF to unify filtering as needed."""
    for v in vcf:
        # Filters
        if biallelic:
            if len(v.ALT) > 1:
                continue

        if snp:
            if not all(len(allele) == 1 for allele in [v.REF] + v.ALT):
                continue

        yield v


def parse_vcf_genotypes(genotypes, format="additive"):
    if format == "additive":
        return _parse_vcf_genotypes_additive(genotypes)
    else:
        raise ValueError(format)


def _parse_vcf_genotypes_additive(genotypes):
    return np.fromiter(
        (a + b for a, b, _ in genotypes),
        dtype=np.float32,
        count=len(genotypes)
    )

# class VCFDataLoader(pl.LightningDataModule):
#     def __init__(self, vcf_filename, batch_size=32):
#         super().__init__()
#         self.vcf_filename = vcf_filename
#         self.batch_size = batch_size
# 
#     def setup(self):
#         pass
# 
#     def train_dataloader(self):
#         return DataLoader()
