#!/usr/bin/env python3
"""
CMSV: Deep Learning-based Structural Variant Detection Tool
PyTorch Version

This script is the main entry point for the CMSV tool.
It supports two modes:
- generate: Generate feature data from BAM files
- call: Call SVs using the trained model

Usage:
    # Generate data
    python CMSV.py generate <bamfile_path_long> <output_data_folder> [max_work] [includecontig]
    
    # Call SVs
    python CMSV.py call <predict_weight> <datapath> <bamfilepath> <predict_path> <outvcfpath> [thread] [includecontig] [num_gpus]
"""

import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional

import torch

from cmsv_features import DEFAULT_FEATURE_VERSION, get_feature_spec, resolve_feature_config
from cmsv_generate_data import create_data_long
from cmsv_detect import (
    model_predict,
    cluster_by_predict,
    collect_prediction_jobs,
    summarize_prediction_jobs,
)
from cmsv_platform import resolve_platform_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DECISION_THRESHOLD = 0.5


def parse_contig_string(contig_str: str) -> List[str]:
    """Parse contig string from command line argument"""
    if not contig_str or contig_str.strip() == '[]':
        return []
    
    contig_str = contig_str.strip('[]')
    contig_list = contig_str.split(',')
    contig_list = [item.strip().strip("'").strip('"') for item in contig_list]
    
    return [c for c in contig_list if c]


def setup_gpu(gpu_id: str = "0"):
    """Setup GPU device(s)"""
    if gpu_id.lower() == 'all':
        # Use all available GPUs
        gpu_count = torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(gpu_count)))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        logger.info(f"Using {gpu_count} GPU(s)")
        for i in range(gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    return device


def resolve_gpu_argument(num_gpus: Optional[str], gpu_override: Optional[str]) -> str:
    """Resolve the original positional num_gpus argument into a CUDA device list."""
    if gpu_override is not None and str(gpu_override).strip():
        return str(gpu_override).strip()

    if num_gpus is None:
        return 'all'

    requested = str(num_gpus).strip()
    if not requested:
        return 'all'
    if requested.lower() == 'all':
        return 'all'
    if ',' in requested:
        return requested

    try:
        gpu_count = int(requested)
    except ValueError as exc:
        raise ValueError(
            f"num_gpus must be an integer count, 'all', or a comma-separated GPU id list; got {num_gpus!r}"
        ) from exc

    if gpu_count <= 0:
        raise ValueError(f"num_gpus must be >= 1, got {gpu_count}")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count > 0:
        gpu_count = min(gpu_count, visible_gpu_count)
    return ','.join(str(i) for i in range(gpu_count))


def resolve_decision_threshold(weights_path: Optional[str], override: Optional[float]) -> tuple[float, str]:
    """Resolve the decision threshold for inference from CLI override or model metadata."""
    if override is not None:
        threshold = float(override)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"decision_threshold must be in [0, 1], got {threshold}")
        return threshold, 'cli_override'

    if weights_path:
        meta_path = Path(weights_path).with_suffix('.meta.json')
    else:
        meta_path = None
    if meta_path and meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            threshold = float(meta.get('decision_threshold', DEFAULT_DECISION_THRESHOLD))
            if 0.0 <= threshold <= 1.0:
                return threshold, str(meta_path)
            logger.warning("Ignoring out-of-range decision threshold %.4f in %s", threshold, meta_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("Failed to read decision threshold metadata from %s", meta_path)

    return DEFAULT_DECISION_THRESHOLD, 'default_0.5'


def resolve_predict_head(weights_path: Optional[str], override: Optional[str]) -> tuple[str, str]:
    """Resolve inference prediction head from CLI override or model metadata."""
    valid_heads = {'cnn', 'mamba', 'cnn_mamba'}
    if override is not None and str(override).strip() and str(override).strip().lower() != 'auto':
        head = str(override).strip().lower()
        if head not in valid_heads:
            raise ValueError(
                f"predict_head must be one of: auto, cnn, mamba, cnn_mamba; got {override!r}"
            )
        return head, 'cli_override'

    if weights_path:
        meta_path = Path(weights_path).with_suffix('.meta.json')
    else:
        meta_path = None
    if meta_path and meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            ablation = meta.get('ablation', {})
            head = str(ablation.get('eval_head', '')).strip().lower()
            if head in valid_heads:
                return head, str(meta_path)
            if head:
                logger.warning("Ignoring unsupported eval_head %r in %s", head, meta_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("Failed to read prediction head metadata from %s", meta_path)

    return 'cnn_mamba', 'default_cnn_mamba'


def mode_generate(args):
    """Generate feature data from BAM file"""
    bamfilepath = args.bamfilepath_long
    outputpath = args.output_data_folder
    max_work = int(args.max_work)
    contigs = parse_contig_string(args.includecontig) if args.includecontig else []
    feature_spec = get_feature_spec(DEFAULT_FEATURE_VERSION)
    
    logger.info("=" * 60)
    logger.info("CMSV - Generate Mode")
    logger.info("=" * 60)
    logger.info(f"BAM file: {bamfilepath}")
    logger.info(f"Output path: {outputpath}")
    logger.info(f"Max workers: {max_work}")
    logger.info(f"BAM threads per worker: {args.bam_threads}")
    logger.info(f"BED pre-filter: {args.bed_file or 'disabled'}")
    logger.info("Feature storage format: npy")
    logger.info(f"Feature storage dtype: {args.feature_dtype}")
    logger.info(f"Skip existing region stores: {not args.overwrite}")
    logger.info(
        "Feature version: %s (dim=%d, normalization=%s)",
        feature_spec['feature_version'],
        feature_spec['feature_dim'],
        feature_spec['normalization'],
    )
    
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        logger.info(f"Created output directory: {outputpath}")
    
    if contigs:
        logger.info(f"Processing contigs: {contigs}")
    else:
        logger.info("Processing all contigs in BAM file")
    
    start_time = time.time()
    
    create_data_long(
        bamfile_long_path=bamfilepath,
        outputpath=outputpath,
        contig=contigs,
        window_size=2000,
        threadss=max_work,
        feature_version=DEFAULT_FEATURE_VERSION,
        bed_file=args.bed_file,
        feature_dtype=args.feature_dtype,
        storage_format='npy',
        overwrite=args.overwrite,
        bam_threads=args.bam_threads,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed:.2f} seconds")
    logger.info("=" * 60)


def mode_call(args):
    """Call SVs using trained model"""
    weights_path = args.predict_weight
    datapath = args.datapath
    bamfilepath = args.bamfilepath
    predict_path = args.predict_path
    vcf_path = args.outvcfpath
    threads = int(args.thread)
    contigs = parse_contig_string(args.includecontig) if args.includecontig else []
    decision_threshold, threshold_source = resolve_decision_threshold(weights_path, args.decision_threshold)
    predict_head, predict_head_source = resolve_predict_head(weights_path, args.predict_head)
    feature_config = resolve_feature_config(data_dir=datapath, weights_path=weights_path)
    platform_name = resolve_platform_name(args.platform, bamfilepath, datapath)
    gpu_spec = resolve_gpu_argument(args.num_gpus, args.gpu) if not args.cluster_only else None
    
    logger.info("=" * 60)
    logger.info("CMSV - Call Mode")
    logger.info("=" * 60)
    logger.info(f"Weights: {weights_path}")
    logger.info(f"Data path: {datapath}")
    logger.info(f"BAM file: {bamfilepath}")
    logger.info(f"Predict path: {predict_path}")
    logger.info(f"VCF output: {vcf_path}")
    logger.info(f"Threads: {threads}")
    logger.info(f"Predict batch size: {args.predict_batch_size}")
    logger.info(f"Genotype method: {args.geno_method}")
    logger.info(f"Platform: {platform_name}")
    logger.info(f"Reuse existing predictions: {args.reuse_predict}")
    logger.info(f"Predict-only mode: {args.predict_only}")
    logger.info(f"Cluster-only mode: {args.cluster_only}")
    logger.info("Decision threshold: %.3f (source: %s)", decision_threshold, threshold_source)
    logger.info("Prediction head: %s (source: %s)", predict_head, predict_head_source)
    logger.info(
        "Support calibration: scale=%.3f cap=%s",
        float(args.support_scale),
        args.support_cap if int(args.support_cap) > 0 else "disabled",
    )
    logger.info(
        "INS support calibration: scale=%.3f floor=%s short_len=%d short_bonus=%d long_len=%d long_delta=%d",
        float(args.ins_support_scale),
        args.ins_min_support_floor if int(args.ins_min_support_floor) > 0 else "disabled",
        int(args.ins_short_len),
        int(args.ins_short_support_bonus),
        int(args.ins_long_len),
        int(args.ins_long_support_delta),
    )
    logger.info(
        "INV support calibration: scale=%.3f floor=%s short_len=%d short_bonus=%d",
        float(args.inv_support_scale),
        args.inv_min_support_floor if int(args.inv_min_support_floor) > 0 else "disabled",
        int(args.inv_short_len),
        int(args.inv_short_support_bonus),
    )
    logger.info("INS eps bands override: %s", args.ins_eps_bands or "default")
    logger.info("INV eps bands override: %s", args.inv_eps_bands or "default")
    logger.info(
        "Feature config: version=%s dim=%d window=%d (source=%s)",
        feature_config['feature_version'],
        feature_config['feature_dim'],
        feature_config['window_size'],
        feature_config['source'],
    )
    logger.info(f"Max DUP/INV length cap: {args.max_dup_inv_len}")
    logger.info(f"Breakpoint refine: {args.bp_refine}")
    if args.bp_refine:
        logger.info(
            "  INS override (min_evidence/max_shift/min_len_ratio/max_len_ratio): %s/%s/%s/%s",
            str(args.bp_refine_ins_min_evidence),
            str(args.bp_refine_ins_max_shift),
            str(args.bp_refine_ins_min_len_ratio),
            str(args.bp_refine_ins_max_len_ratio),
        )
    
    # Setup device
    device = None
    if not args.cluster_only:
        device = setup_gpu(gpu_spec)
    
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
        logger.info(f"Created predict directory: {predict_path}")
    
    if not os.path.exists(vcf_path):
        os.makedirs(vcf_path)
        logger.info(f"Created VCF directory: {vcf_path}")
    
    if contigs:
        logger.info(f"Processing contigs: {contigs}")
    else:
        logger.info("Processing all contigs in BAM file")
    
    start_time = time.time()
    
    # Step 1: Model prediction
    logger.info("\n--- Step 1: Model Prediction ---")
    if args.cluster_only:
        ready_jobs, missing_jobs, empty_store_count = summarize_prediction_jobs(
            collect_prediction_jobs(
                bamfilepath=bamfilepath,
                data_path=datapath,
                testpath=predict_path,
                contigg=contigs,
            )[2]
        )
        logger.info(
            "Cluster-only mode: reusing %d prediction regions, %d missing predictions ignored, %d empty stores skipped",
            len(ready_jobs),
            len(missing_jobs),
            empty_store_count,
        )
    else:
        skip_prediction = False
        if args.reuse_predict:
            ready_jobs, missing_jobs, empty_store_count = summarize_prediction_jobs(
                collect_prediction_jobs(
                    bamfilepath=bamfilepath,
                    data_path=datapath,
                    testpath=predict_path,
                    contigg=contigs,
                )[2]
            )
            if not missing_jobs:
                skip_prediction = True
                logger.info(
                    "All %d non-empty prediction regions already exist under %s (%d empty stores skipped)",
                    len(ready_jobs),
                    predict_path,
                    empty_store_count,
                )
            else:
                logger.info(
                    "Prediction reuse enabled: %d regions ready, %d regions still need forward pass, %d empty stores skipped",
                    len(ready_jobs),
                    len(missing_jobs),
                    empty_store_count,
                )

        if skip_prediction:
            logger.info("Skipping model prediction because reusable predictions are already complete")
        else:
            model_predict(
                weights_path=weights_path,
                bamfilepath=bamfilepath,
                data_path=datapath,
                testpath=predict_path,
                contigg=contigs,
                device=str(device),
                model_type=args.model_type,
                platform=platform_name,
                predict_batch_size=args.predict_batch_size,
                predict_head=predict_head,
                reuse_existing=args.reuse_predict,
            )
            logger.info("Prediction completed")

    if args.predict_only:
        elapsed = time.time() - start_time
        logger.info("Predict-only mode: skipping clustering and VCF generation")
        logger.info(f"\nCompleted in {elapsed:.2f} seconds")
        logger.info("=" * 60)
        return
    
    # Step 2: Clustering and VCF generation
    logger.info("\n--- Step 2: Clustering and VCF Generation ---")
    cluster_by_predict(
        bamfilepath=bamfilepath,
        data_path=datapath,
        testpath=predict_path,
        outputpath=vcf_path,
        contigg=contigs,
        threads_numm=threads,
        decision_threshold=decision_threshold,
        geno_method=args.geno_method,
        max_dup_inv_len=args.max_dup_inv_len,
        support_scale=args.support_scale,
        support_cap=args.support_cap,
        ins_support_scale=args.ins_support_scale,
        ins_min_support_floor=args.ins_min_support_floor,
        ins_short_len=args.ins_short_len,
        ins_short_support_bonus=args.ins_short_support_bonus,
        ins_long_len=args.ins_long_len,
        ins_long_support_delta=args.ins_long_support_delta,
        inv_support_scale=args.inv_support_scale,
        inv_min_support_floor=args.inv_min_support_floor,
        inv_short_len=args.inv_short_len,
        inv_short_support_bonus=args.inv_short_support_bonus,
        ins_eps_bands=args.ins_eps_bands,
        inv_eps_bands=args.inv_eps_bands,
        bp_refine=args.bp_refine,
        bp_refine_window=args.bp_refine_window,
        bp_refine_min_evidence=args.bp_refine_min_evidence,
        bp_refine_max_shift=args.bp_refine_max_shift,
        bp_refine_types=args.bp_refine_types,
        bp_refine_ins_min_evidence=args.bp_refine_ins_min_evidence,
        bp_refine_ins_max_shift=args.bp_refine_ins_max_shift,
        bp_refine_ins_min_len_ratio=args.bp_refine_ins_min_len_ratio,
        bp_refine_ins_max_len_ratio=args.bp_refine_ins_max_len_ratio,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed:.2f} seconds")
    logger.info("=" * 60)


def _choose_arg(
    parser: argparse.ArgumentParser,
    positional_value,
    flag_value,
    default,
    name: str,
):
    if positional_value is not None:
        return positional_value
    if flag_value is not None:
        return flag_value
    if default is not None:
        return default
    parser.error(f"{name} is required")


def finalize_generate_args(parser: argparse.ArgumentParser, args) -> None:
    args.bamfilepath_long = _choose_arg(
        parser, args.bamfilepath_long, args.bamfile_flag, None, 'bamfile_path_long'
    )
    args.output_data_folder = _choose_arg(
        parser, args.output_data_folder, args.output_flag, None, 'output_data_folder'
    )
    args.max_work = int(_choose_arg(parser, args.max_work, args.threads_flag, 4, 'max_work'))
    args.includecontig = _choose_arg(parser, args.includecontig, args.contigs_flag, '[]', 'includecontig')


def finalize_call_args(parser: argparse.ArgumentParser, args) -> None:
    if args.predict_only and args.cluster_only:
        parser.error("--predict_only and --cluster_only are mutually exclusive")
    if args.cluster_only:
        args.predict_weight = args.predict_weight or args.weights_flag
    else:
        args.predict_weight = _choose_arg(
            parser, args.predict_weight, args.weights_flag, None, 'predict_weight'
        )
    args.datapath = _choose_arg(parser, args.datapath, args.datapath_flag, None, 'datapath')
    args.bamfilepath = _choose_arg(parser, args.bamfilepath, args.bamfile_flag, None, 'bamfilepath')
    args.predict_path = _choose_arg(parser, args.predict_path, args.predict_path_flag, None, 'predict_path')
    args.outvcfpath = _choose_arg(parser, args.outvcfpath, args.vcf_path_flag, None, 'outvcfpath')
    args.thread = int(_choose_arg(parser, args.thread, args.threads_flag, 15, 'thread'))
    args.includecontig = _choose_arg(parser, args.includecontig, args.contigs_flag, '[]', 'includecontig')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CMSV: Deep Learning-based Structural Variant Detection Tool (PyTorch Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate feature data
  python CMSV.py generate input.bam ./data 15
  
  # Generate with specific contigs
  python CMSV.py generate input.bam ./data 15 "[chr1,chr2,chr3]"
  
  # Call SVs
  python CMSV.py call model.pth ./data input.bam ./predict ./vcf 15
  
  # Call with GPU selection
  python CMSV.py call model.pth ./data input.bam ./predict ./vcf 15 [] 2

  # Call with log-space genotype strategy
  python CMSV.py call model.pth ./data input.bam ./predict ./vcf --geno_method loggl

  # Call with threshold auto-loaded from model metadata
  python CMSV.py call best_model.pth ./data input.bam ./predict ./vcf

  # Call with explicit platform override
  python CMSV.py call best_model.pth ./data input.bam ./predict ./vcf --platform ont

  # Call with DUP/INV length cap (bp)
  python CMSV.py call model.pth ./data input.bam ./predict ./vcf --max_dup_inv_len 2000000

  # Call with breakpoint refinement
  python CMSV.py call model.pth ./data input.bam ./predict ./vcf --bp_refine

  # Reuse existing predictions and only run missing forward passes
  python CMSV.py call model.pth ./data input.bam ./predict ./vcf --reuse_predict

  # Cluster-only rerun using existing predict/index files
  python CMSV.py call --cluster_only -d ./data -b input.bam -p ./predict -v ./vcf --decision_threshold 0.55
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    # Generate subparser
    generate_parser = subparsers.add_parser('generate', help='Generate feature data from BAM file')
    generate_parser.add_argument('bamfilepath_long', nargs='?', help='Input BAM file path')
    generate_parser.add_argument('output_data_folder', nargs='?', help='Output directory for feature data')
    generate_parser.add_argument('max_work', nargs='?', type=int, default=None, help='Number of region worker processes (default: 4)')
    generate_parser.add_argument('includecontig', nargs='?', default=None, help='Contig list string, e.g. [] or [chr1,chr2]')
    generate_parser.add_argument('-b', '--bamfile', dest='bamfile_flag', default=None, help=argparse.SUPPRESS)
    generate_parser.add_argument('-o', '--output', dest='output_flag', default=None, help=argparse.SUPPRESS)
    generate_parser.add_argument('-t', '--threads', dest='threads_flag', type=int, default=None, help=argparse.SUPPRESS)
    generate_parser.add_argument('-c', '--contigs', dest='contigs_flag', default=None, help=argparse.SUPPRESS)
    generate_parser.add_argument('--bed_file', default=None, help='Optional BED file used to pre-filter generated windows')
    generate_parser.add_argument('--bam_threads', type=int, default=2, help='Per-worker BAM decode threads during feature generation (default: 2)')
    generate_parser.add_argument('--feature_dtype', default='float16', choices=['float16', 'float32'], help='Storage dtype for generated feature windows')
    generate_parser.add_argument('--overwrite', action='store_true', help='Recompute and overwrite existing region stores instead of resuming')
    
    # Call subparser
    call_parser = subparsers.add_parser('call', help='Call SVs using trained model')
    call_parser.add_argument('predict_weight', nargs='?', help='Path to model weights file (optional with --cluster_only)')
    call_parser.add_argument('datapath', nargs='?', help='Path to feature data directory')
    call_parser.add_argument('bamfilepath', nargs='?', help='Input BAM file path')
    call_parser.add_argument('predict_path', nargs='?', help='Path to save predictions')
    call_parser.add_argument('outvcfpath', nargs='?', help='Path to save VCF output')
    call_parser.add_argument('thread', nargs='?', type=int, default=None, help='Number of worker threads (default: 15)')
    call_parser.add_argument('includecontig', nargs='?', default=None, help='Contig list string, e.g. [] or [chr1,chr2]')
    call_parser.add_argument('num_gpus', nargs='?', default=None, help='Optional GPU count, "all", or comma-separated GPU ids')
    call_parser.add_argument('-w', '--weights', dest='weights_flag', default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('-d', '--datapath', dest='datapath_flag', default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('-b', '--bamfile', dest='bamfile_flag', default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('-p', '--predict_path', dest='predict_path_flag', default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('-v', '--vcf_path', dest='vcf_path_flag', default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('-t', '--threads', dest='threads_flag', type=int, default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('-c', '--contigs', dest='contigs_flag', default=None, help=argparse.SUPPRESS)
    call_parser.add_argument('--model_type', type=str, default='mamba', choices=['mamba'], help='Model architecture type (only mamba is supported)')
    call_parser.add_argument('--geno_method', type=str, default='loggl', choices=['vote', 'loggl'], help='Genotype strategy: vote (legacy) or loggl (default, log-space likelihood)')
    call_parser.add_argument('--decision_threshold', type=float, default=None, help='Decision threshold override (default: auto-load from <weights>.meta.json, else 0.5)')
    call_parser.add_argument('--max_dup_inv_len', type=int, default=2000000, help='Maximum allowed DUP/INV length in bp (0 to disable cap)')
    call_parser.add_argument('--support_scale', type=float, default=1.0, help='Scale inferred min_support at inference time (default: 1.0)')
    call_parser.add_argument('--support_cap', type=int, default=0, help='Optional hard cap for inferred min_support at inference time (0 disables cap)')
    call_parser.add_argument('--ins_support_scale', type=float, default=1.0, help='INS-specific scale for inferred min_support (default: 1.0)')
    call_parser.add_argument('--ins_min_support_floor', type=int, default=0, help='INS-specific minimum support floor after scaling (0 disables)')
    call_parser.add_argument('--ins_short_len', type=int, default=300, help='INS length threshold for short-INS support tightening (bp)')
    call_parser.add_argument('--ins_short_support_bonus', type=int, default=0, help='Extra min_support added to INS with SVLEN <= --ins_short_len')
    call_parser.add_argument('--ins_long_len', type=int, default=0, help='INS length threshold for long-INS support adjustment (0 disables)')
    call_parser.add_argument('--ins_long_support_delta', type=int, default=0, help='Additive support adjustment for INS with SVLEN >= --ins_long_len (can be negative)')
    call_parser.add_argument('--inv_support_scale', type=float, default=1.0, help='INV-specific scale for inferred min_support (default: 1.0)')
    call_parser.add_argument('--inv_min_support_floor', type=int, default=0, help='INV-specific minimum support floor after scaling (0 disables)')
    call_parser.add_argument('--inv_short_len', type=int, default=3000, help='INV length threshold for short-INV support tightening (bp)')
    call_parser.add_argument('--inv_short_support_bonus', type=int, default=0, help='Extra min_support added to INV with SVLEN <= --inv_short_len')
    call_parser.add_argument('--ins_eps_bands', type=str, default=None, help='Override INS length-aware DBSCAN eps bands as max_len:eps,...')
    call_parser.add_argument('--inv_eps_bands', type=str, default=None, help='Override INV length-aware DBSCAN eps bands as max_len:eps,...')
    call_parser.add_argument('--platform', type=str, default=None, help='Sequencing platform override: ccs, clr, or ont (default: auto-infer from BAM/data path)')
    call_parser.add_argument('--gpu', default=None, help='Override positional num_gpus with comma-separated GPU ids or "all"')
    call_parser.add_argument('--predict_batch_size', type=int, default=64, help='Inference batch size per forward pass (default: 64)')
    call_parser.add_argument('--predict_head', choices=['auto', 'cnn', 'mamba', 'cnn_mamba'], default='auto', help='Prediction head for model forward pass; auto-load from <weights>.meta.json, else default to cnn_mamba')
    call_parser.add_argument('--reuse_predict', action='store_true', help='Reuse existing region prediction files under --predict_path and only run missing forward passes')
    call_parser.add_argument('--predict_only', action='store_true', help='Run only the model forward pass and write predict/*.npy without clustering or VCF generation')
    call_parser.add_argument('--cluster_only', action='store_true', help='Skip model forward pass and rerun clustering/post-processing from existing predict + index files')
    call_parser.add_argument('--bp_refine', action='store_true', help='Enable breakpoint refinement post-processing')
    call_parser.add_argument('--bp_refine_window', type=int, default=200, help='Refinement search window around breakpoints (bp)')
    call_parser.add_argument('--bp_refine_min_evidence', type=int, default=2, help='Minimum evidence reads to apply refinement')
    call_parser.add_argument('--bp_refine_max_shift', type=int, default=500, help='Maximum allowed breakpoint shift during refinement (bp)')
    call_parser.add_argument('--bp_refine_types', type=str, default='DEL', help='Comma-separated SV types for refinement (default: DEL)')
    call_parser.add_argument('--bp_refine_ins_min_evidence', type=int, default=None, help='INS-specific minimum evidence reads (fallback to --bp_refine_min_evidence)')
    call_parser.add_argument('--bp_refine_ins_max_shift', type=int, default=None, help='INS-specific max breakpoint shift in bp (fallback to --bp_refine_max_shift)')
    call_parser.add_argument('--bp_refine_ins_min_len_ratio', type=float, default=None, help='INS-specific minimum refined/original SVLEN ratio (fallback to 0.5)')
    call_parser.add_argument('--bp_refine_ins_max_len_ratio', type=float, default=None, help='INS-specific maximum refined/original SVLEN ratio (fallback to 2.0)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'generate':
        finalize_generate_args(parser, args)
        mode_generate(args)
    elif args.mode == 'call':
        finalize_call_args(parser, args)
        mode_call(args)
    else:
        parser.error(f"Unsupported mode: {args.mode}")


if __name__ == '__main__':
    main()
