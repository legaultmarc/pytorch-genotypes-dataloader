from dataclasses import dataclass
import collections
import typing
import sqlite3
import os

import cyvcf2
import numpy as np
import torch


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

    _Chunk = collections.namedtuple("_Chunk", ("id", "chrom", "start", "end",
                                               "n_variants"))

    def __init__(self, vcf_filename, max_snps_per_chunk=200000, create=False):
        self.vcf_filename = vcf_filename

        db_filename = os.path.abspath(self._get_db_name())
        print(f"Using DB: {db_filename}")
        self.con = sqlite3.connect(db_filename)
        if create:
            self.create(max_snps_per_chunk)

        # Load the chunk regions in memory.
        self._load_chunks()

    def _load_chunks(self):
        cur = self.con.cursor()

        cur.execute(
            "select chunks.*, counts.n "
            "from "
            "  chunks inner join "
            "  ( "
            "       select chunk_id, count(*) as n "
            "       from variants group by chunk_id "
            "  ) counts on chunks.id=counts.chunk_id;"
        )
        self.chunks = [self._Chunk(*tu) for tu in cur.fetchall()]

    def get_samples(self):
        try:
            vcf = cyvcf2.VCF(self.vcf_filename)
            samples = vcf.samples
        finally:
            vcf.close()

        return samples

    def get_chunk_meta(self, chunk_id):
        li = filter(lambda chunk: chunk.id == chunk_id, self.chunks)
        li = list(li)
        if len(li) > 1:
            raise ValueError()
        elif len(li) == 0:
            raise IndexError(chunk_id)

        return li[0]

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
            "  id text, "
            "  pos integer not null, "
            "  ref text not null, "
            "  alt text not null, "
            "  constraint chunk_fk "
            "    foreign key (chunk_id) references chunks (id)"
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

            buf.append([cur_chunk.chunk_id, v.CHROM, v.ID, v.POS,
                        v.REF, v.ALT[0]])
            cur_n += 1

            if len(buf) >= 1e6:
                buf = self._flush_buffer(buf)

            prev = v

        if buf:
            self._flush_buffer(buf)

        self._close_chunk(cur_chunk, last_variant=v)

    def _flush_buffer(self, buf):
        cur = self.con.cursor()
        cur.executemany("insert into variants values (?, ?, ?, ?, ?, ?)", buf)
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
        chunk = self.get_chunk_meta(chunk_id)
        return iter_vcf_wrapper(
            cyvcf2.VCF(self.vcf_filename)(
                f"{chunk.chrom}:{chunk.start}-{chunk.end}"
            )
        )

    def get_n_chunks(self):
        return len(self.chunks)

    def get_variant_metadata_for_chunk_id(self, chunk_id):
        import pandas as pd
        cur = self.con.cursor()

        cur.execute(
            "select chrom, id, pos, ref, alt "
            "  from variants where chunk_id=? order by pos asc;",
            (chunk_id, )
        )

        results = cur.fetchall()
        if not results:
            raise ValueError(f"No variants in chunk '{chunk_id}'.")

        return pd.DataFrame(results, columns=["chrom", "id", "pos",
                                              "ref", "alt"])

    def get_tensor_for_chunk_id(self, chunk_id):
        # Check how many samples and variants to pre-allocate memory.
        try:
            vcf = cyvcf2.VCF(self.vcf_filename)
            n_samples = len(vcf.samples)
        finally:
            vcf.close()

        chunk = self.get_chunk_meta(chunk_id)
        mat = np.empty((n_samples, chunk.n_variants), dtype=np.float32)

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
        (a + b if a != -1 and b != -1 else np.nan
         for a, b, _ in genotypes),
        dtype=np.float32,
        count=len(genotypes)
    )
