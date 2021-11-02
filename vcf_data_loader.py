from dataclasses import dataclass
import typing

import sqlite3
import cyvcf2
import gzip
from torch.utils.data import DataLoader
import pytorch_lightning as pl


@dataclass
class VCFChunk:
    chunk_id: int
    chrom: str
    start: int
    end: int

    @classmethod
    def new(cls, id):
        return cls(id, None, None, None)

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

        self.con = sqlite3.connect("vcf_chunks.db")
        if create:
            self.create(max_snps_per_chunk)

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
        cur_chunk = VCFChunk.new(0)
        cur_n = 0
        buf = []
        for v in cyvcf2.VCF(self.vcf_filename, lazy=True):
            if cur_chunk.chrom is None:
                cur_chunk.chrom = v.CHROM
                cur_chunk.start = v.POS

            if cur_chunk.chrom != v.CHROM or cur_n >= max_snps_per_chunk:
                cur_chunk = self._close_chunk(v, cur_chunk)
                cur_n = 0

            # Only consider bi-allelic variants.
            if len(v.ALT) > 1:
                continue

            alt = v.ALT[0]

            buf.append([cur_chunk.chunk_id, v.CHROM, v.POS, v.REF, alt])
            cur_n += 1

            if len(buf) >= 1e6:
                buf = self._flush_buffer(buf)

    def _flush_buffer(self, buf):
        cur = self.con.cursor()
        cur.executemany("insert into variants values (?, ?, ?, ?, ?)", buf)
        return []

    def _close_chunk(self, v, cur_chunk: VCFChunk):
        # Increment chunk counter and add entry to the db.
        cur_chunk.end = v.POS
        cur = self.con.cursor()
        cur_chunk.sql_insert(cur)
        self.con.commit()

        return VCFChunk.new(cur_chunk.chunk_id + 1)

    def iter_vcf_by_chunk(
        self,
        chunk: VCFChunk
    ) -> typing.Generator[cyvcf2.Variant, None, None]:
        return self.iter_vcf_by_chunk_id(chunk.chunk_id)

    def iter_vcf_by_chunk_id(
        self,
        chunk_id: int
    ) -> typing.Generator[cyvcf2.Variant, None, None]:
        cur = self.con.cursor()
        cur.execute(
            "select chrom, start, end from chunks where id=?", (chunk_id, )
        )
        chrom, start, end = cur.fetchone()
        return cyvcf2.VCF(self.vcf_filename)(f"{chrom}:{start}-{end}")

    def get_n_chunks(self):
        cur = self.con.cursor()
        cur.execute("select count(*) from chunks")
        return cur.fetchone()[0]


class VCFDataLoader(pl.LightningDataModule):
    def __init__(self, vcf_filename, batch_size=32):
        super().__init__()
        self.vcf_filename = vcf_filename
        self.batch_size = batch_size

    def setup(self):
        pass

    def train_dataloader(self):
        return DataLoader()
