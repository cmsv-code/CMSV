# CMSV

CMSV is an inference-only structural variant caller for long-read sequencing data.

CMSV is intended to run with an NVIDIA GPU. CPU-only inference is not supported in this release.

This repository contains the minimal code and pretrained weight needed to run:

```text
BAM -> feature generation -> model prediction -> clustering -> VCF
```

Training code, training data, benchmarks, logs, and intermediate outputs are not included.

## Contents

```text
CMSV.py
cmsv_detect.py
cmsv_features.py
cmsv_generate_data.py
cmsv_model.py
cmsv_platform.py
cmsv_storage.py
weights/
  cmsv_cnn_mamba_main_best.pth
  cmsv_cnn_mamba_main_best.meta.json
requirements.txt
environment_cmsv.yml
```

## Environment

Create the conda environment:

```bash
conda env create -f environment_cmsv.yml
conda activate cmsv
```

`mamba env create -f environment_cmsv.yml` can be used as a faster drop-in replacement for `conda env create`.

CMSV requires `mamba-ssm`, `causal-conv1d`, and a CUDA-enabled PyTorch build compatible with your NVIDIA driver.

## Input Requirements

CMSV needs:

- A coordinate-sorted BAM file.
- A BAM index file beside it, usually `input.bam.bai`.
- A reference-compatible contig naming convention, for example `chr1` to `chr22`.

## Run Inference

Generate features:

```bash
python CMSV.py generate \
  input.bam \
  ./work \
  4 \
  "[chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22]" \
  --bam_threads 1 \
  --feature_dtype float16
```

Call SVs on GPU:

```bash
python CMSV.py call \
  ./weights/cmsv_cnn_mamba_main_best.pth \
  ./work \
  input.bam \
  ./predict \
  ./vcf \
  8 \
  "[chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22]" \
  --predict_head cnn_mamba \
  --decision_threshold 0.88 \
  --predict_batch_size 128 \
  --reuse_predict
```

Set the platform explicitly when needed:

```bash
--platform ccs
--platform clr
--platform ont
```

The final VCF files are written under `./vcf/<sample_name>/`.

## Resume Modes

Reuse existing predictions and only run missing forward passes:

```bash
python CMSV.py call \
  ./weights/cmsv_cnn_mamba_main_best.pth \
  ./work input.bam ./predict ./vcf 8 \
  "[chr1,chr2,chr3]" \
  --reuse_predict
```

Skip model inference and rerun clustering/VCF generation from existing predictions:

```bash
python CMSV.py call \
  --cluster_only \
  ./weights/cmsv_cnn_mamba_main_best.pth \
  ./work input.bam ./predict ./vcf 8 \
  "[chr1,chr2,chr3]" \
  --decision_threshold 0.88
```

## Pretrained Weight

The included weight is:

```text
weights/cmsv_cnn_mamba_main_best.pth
```

It is the CNN+Mamba main-head model used for the current CMSV inference workflow. The paired metadata file stores the model head and threshold information used during validation.
