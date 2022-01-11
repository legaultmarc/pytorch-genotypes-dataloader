import vcf_data_loader

def compute_ld(data):
    n = data.size()[0]
    # Standardize
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    return (data.transpose(0, 1) @ data / n) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vcf = vcf_data_loader.FixedSizeVCFChunks(
        "all_1kg_chr1_phased_GRCh38_snps_maf0.01.recode.vcf.gz"
    )

    data = vcf.get_dataset_for_chunk_id(20)
    ld = compute_ld(data)

    plt.matshow(ld)
    plt.colorbar()
    plt.show()