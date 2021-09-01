#!/usr/bin/env python


"""
Script to convert genotypes into the scipy sparse matrix format.
"""

import os
import argparse
import geneparse
import pandas as pd
import numpy as np
import scipy.sparse
import uuid


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("genotypes", type=str)
    parser.add_argument("--format", "-f", type=str)
    parser.add_argument("--kwargs", type=str, default=None)
    parser.add_argument("--n-variants", type=int, default=None)
    parser.add_argument("--maf", type=int, default=0.01)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def create_sparse_mat(triplets, m, n, dtype=np.int8):
    triplets = np.array(triplets)
    return scipy.sparse.coo_matrix(
        (triplets[:, 2], (triplets[:, 0], triplets[:, 1])),
        shape=(m, n),
        dtype=dtype
    )


def checkpoint(triplets, m, n, seed, chunks):
    cur = create_sparse_mat(triplets, m, n)
    filename = f"{seed}_{len(chunks)}.npz"
    scipy.sparse.save_npz(filename, cur)
    chunks.append(filename)
    return chunks


def main():
    args = parse_args()

    seed = str(uuid.uuid4()).split("-")[0]

    triplets = []
    variants = []

    kwargs = {}
    if args.kwargs is not None:
        for token in args.kwargs.split(";"):
            for k, v in token.split("="):
                kwargs[k] = v

    chunks = []
    MAX_TRIPLETS = 5e6

    n_included_variants = 0
    target_n_variants = args.n_variants if args.n_variants is not None else n

    with geneparse.parsers[args.format](args.genotypes, **kwargs) as reader:
        samples = reader.get_samples()
        m = len(samples)
        n = reader.get_number_variants()

        for j, g in enumerate(reader.iter_genotypes()):
            freq = g.coded_freq()
            if freq > 0.5:
                # Code minor (TODO: Make sure coding is ok when implementing).
                g = g.flip()
                maf = 1 - freq
            else:
                maf = freq

            if maf < args.maf:
                continue

            n_included_variants += 1
            variants.append(g.variant)

            vec = np.round(g.genotypes)
            triplets.extend([
                (i, j, v) for i, v
                in enumerate(g.genotypes)
                if np.round(v) != 0
            ])

            if n_included_variants >= target_n_variants:
                if args.debug:
                    print(f"Stopping at {args.n_variants} variants.")

                break

            if len(triplets) > MAX_TRIPLETS:
                chunks = checkpoint(
                    triplets, m, target_n_variants, seed, chunks
                )
                triplets = []

            if args.debug and (j % 100000) == 0:
                print(j)

    if triplets:
        chunks = checkpoint(triplets, m, target_n_variants, seed, chunks)

    del triplets

    if n_included_variants != target_n_variants:
        if args.debug:
            print(n_included_variants, target_n_variants)

    print("Collapsing chunks")
    out = scipy.sparse.vstack([
        scipy.sparse.load_npz(filename) for filename in chunks
    ], dtype=np.int8)

    for filename in chunks:
        os.remove(filename)

    print("Writing output")
    scipy.sparse.save_npz("output.npz", out)


if __name__ == "__main__":
    main()
