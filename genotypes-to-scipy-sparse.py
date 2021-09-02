#!/usr/bin/env python


"""
Script to convert genotypes into the scipy sparse matrix format.
"""

import os
import pickle
import gzip
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

    n_included_variants = 0
    target_n_variants = args.n_variants if args.n_variants is not None else n

    out = None
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
            mask = np.where(vec != 0)[0]

            col = scipy.sparse.coo_matrix(
                (vec[mask], (mask, np.zeros(mask.shape[0]))),
                shape=(m, 1),
                dtype=np.int8
            )

            if out is None:
                out = col
            else:
                out = scipy.sparse.hstack((out, col))

            if n_included_variants >= target_n_variants:
                if args.debug:
                    print(f"Stopping at {args.n_variants} variants.")

                break

            if args.debug and (j % 100000) == 0:
                print(j)

    print("Writing output")
    with gzip.open("output_variants.pkl.gz", "wb") as f:
        pickle.dump(variants, f)

    with gzip.open("output_samples.txt.gz", "wt") as f:
        for s in samples:
            f.write(str(s) + "\n")

    scipy.sparse.save_npz("output_matrix.npz", out)


if __name__ == "__main__":
    main()
