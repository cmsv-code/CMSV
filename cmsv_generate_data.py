"""
CMSV Data Generation Module (PyTorch Version)
Generates feature data from BAM files for SV detection
"""

import pysam
import pandas as pd
import numpy as np
import math
import os
import time
import gc
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional
from statistics import mean
import logging

from cmsv_features import (
    DEFAULT_FEATURE_VERSION,
    get_feature_spec,
    write_feature_config,
)
from cmsv_storage import (
    feature_store_path,
    feature_store_complete,
    save_npy_feature_store,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


REGION_SIZE = 10_000_000
_GENERATE_WORKER_BAM = None
_GENERATE_WORKER_BAM_PATH = None
GENERATE_SUBREGION_SIZE_ENV = "VARIANTCALLER_GENERATE_SUBREGION_SIZE"


# ============================================================================
# Signal Detection
# ============================================================================

SIGNAL_FLAGS = {
    1 << 2: 0,
    1 >> 1: 1,
    1 << 4: 2,
    1 << 11: 3,
    1 << 4 | 1 << 11: 4
}


def detect_flag(flag: int) -> int:
    """Detect read flag signal"""
    return SIGNAL_FLAGS.get(flag, 0)


# ============================================================================
# CIGAR Parsing Functions
# ============================================================================

def c_pos(cigar: str, refstart: int) -> Tuple[int, int, int, int]:
    """Parse CIGAR string to get reference and read positions"""
    number = ''
    readstart = None
    readend = None
    refend = None
    readloc = 0
    refloc = refstart

    for c in cigar:
        if c.isdigit():
            number += c
        else:
            number = int(number)
            if c in ['S', 'H'] and readstart is None:
                if c == 'S':
                    readstart = readloc + number
                readloc += number
            else:
                if readstart is None and c in ['M', 'I', '=', 'X']:
                    readstart = readloc
                if c in ['M', 'I', '=', 'X']:
                    readloc += number
                if c in ['M', 'D', 'N', '=', 'X']:
                    refloc += number
                if c in ['H', 'S'] and readstart is not None:
                    readend = readloc
                    refend = refloc
            number = ''

    if readend is None:
        readend = readloc
    if refend is None:
        refend = refloc

    return refstart, refend, readstart, readend


def splitreadlist(read) -> List[List]:
    """Extract split read information from supplementary alignments"""
    sv_list = []
    aligned_length = read.reference_length
    process_signal = detect_flag(read.flag)
    
    if process_signal in [1, 2]:
        if read.is_reverse:
            qry_start = read.infer_read_length() - read.query_alignment_end
            qry_end = read.infer_read_length() - read.query_alignment_start
        else:
            qry_start = read.query_alignment_start
            qry_end = read.query_alignment_end
        
        qry_reference_start = read.reference_start
        qry_reference_end = read.reference_end
        strand_ = '-' if read.is_reverse else '+'
        sv_list.append([qry_start, qry_end, qry_reference_start, qry_reference_end, read.reference_name, strand_])
        
        rawsalist = read.get_tag('SA').split(';')
        for sa in rawsalist[:-1]:
            sainfo = sa.split(',')
            tmpcontig, tmprefstart, strand, cigar, sup_mapq = (
                sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3], int(sainfo[4])
            )
            refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)

            if strand == '-' and sup_mapq >= 0:
                readstart = read.query_length - readend_2
                readend = read.query_length - readstart_2
                sv_list.append([readstart, readend, refstart_2, refend_2, tmpcontig, strand])
            elif strand == '+' and sup_mapq >= 0:
                sv_list.append([readstart_2, readend_2, refstart_2, refend_2, tmpcontig, strand])

    return sv_list


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def feature_record(alignment_current: List, alignment_next: List) -> Tuple:
    """Calculate features between two aligned segments"""
    distance_on_read = alignment_next[0] - alignment_current[1]
    
    if alignment_current[-1] == '+':
        distance_on_reference = alignment_next[2] - alignment_current[3]
        if alignment_next[-1] == '-':
            if alignment_current[3] > alignment_next[3]:
                distance_on_reference = alignment_next[3] - alignment_current[2]
            else:
                distance_on_reference = alignment_current[3] - alignment_next[2]
    else:
        distance_on_reference = alignment_current[2] - alignment_next[3]
        if alignment_next[-1] == '+':
            if alignment_current[3] > alignment_next[3]:
                distance_on_reference = alignment_next[3] - alignment_current[2]
            else:
                distance_on_reference = alignment_current[3] - alignment_next[2]
    
    deviation = distance_on_read - distance_on_reference
    chr_ = 1 if alignment_current[-2] == alignment_next[-2] else 0
    orientation = 1 if alignment_current[-1] == alignment_next[-1] else 0
    
    return (alignment_current, alignment_next, chr_, orientation, distance_on_read, 
            distance_on_reference, deviation, False)


def feature_read_segment(svlist: List) -> List:
    """Extract features from read segments"""
    sg_list = []
    sorted_alignment_list = sorted(svlist, key=lambda aln: (aln[0], aln[1]))
    
    for index in range(len(sorted_alignment_list) - 1):
        sg_list.append(feature_record(sorted_alignment_list[index], sorted_alignment_list[index + 1]))
    
    if len(svlist) >= 3 and sorted_alignment_list[0][-2] != sorted_alignment_list[1][-2]:
        sg_list.append(feature_record(sorted_alignment_list[0], sorted_alignment_list[-1]))
    
    return sg_list


def analyze_read_segments(read, segment_data: List, candidate: List):
    """Analyze read segments to identify SV candidates"""
    min_sv_size = 40
    segment_overlap_tolerance = 5
    read_name = read.query_name
    
    for sv_sig in segment_data:
        alignment_current = sv_sig[0]
        alignment_next = sv_sig[1]
        ref_chr = alignment_current[-2]
        chr_, orientation, distance_on_read, distance_on_reference, deviation, long_ins = sv_sig[2:]
        
        if chr_ == 1:  # Same chromosome
            if orientation == 1:  # Same orientation
                if distance_on_reference >= -min_sv_size or long_ins:
                    if deviation > 0:  # INS
                        if alignment_current[-1] == '+':
                            start = ((alignment_current[3] + alignment_next[2]) // 2 
                                    if not long_ins else min(alignment_current[3], alignment_next[2]))
                        else:
                            start = ((alignment_current[2] + alignment_next[3]) // 2 
                                    if not long_ins else min(alignment_current[2], alignment_next[3]))
                        end = start + deviation
                        if end - start >= min_sv_size:
                            candidate.append([start, deviation, read_name, 'A', 'INS', ref_chr, read.mapping_quality])
                    
                    elif deviation < 0:  # DEL
                        if alignment_current[-1] == '+':
                            start = alignment_current[3]
                        else:
                            start = alignment_next[3]
                        end = start - deviation
                        if end - start >= min_sv_size:
                            candidate.append([start, -deviation, read_name, 'None', 'DEL', ref_chr, read.mapping_quality])
                else:  # DUP
                    if alignment_current[-1] == '+':
                        start = alignment_next[2]
                        end = alignment_current[3]
                    else:
                        start = alignment_current[2]
                        end = alignment_next[3]
                    svlen = end - start
                    candidate.append([start, svlen, read_name, 'None', 'DUP', ref_chr, read.mapping_quality])
            
            else:  # INV
                if alignment_current[-1] == '+':
                    if alignment_next[2] - alignment_current[3] >= -segment_overlap_tolerance:
                        start = alignment_current[3]
                        end = alignment_next[3]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr, read.mapping_quality])
                    elif alignment_current[2] - alignment_next[3] >= -segment_overlap_tolerance:
                        start = alignment_next[3]
                        end = alignment_current[3]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr, read.mapping_quality])
                else:
                    if alignment_next[2] - alignment_current[3] >= -segment_overlap_tolerance:
                        start = alignment_current[2]
                        end = alignment_next[2]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr, read.mapping_quality])
                    elif alignment_current[2] - alignment_next[3] >= -segment_overlap_tolerance:
                        start = alignment_next[2]
                        end = alignment_current[2]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr, read.mapping_quality])
        
        else:  # BND/Translocation
            ref_chr_next = alignment_next[-2]
            if orientation == 1:
                if alignment_current[-1] == '+':
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[3]
                    candidate.append([ref_chr, start, 'fwd', ref_chr_next, end, 'fwd', 'BND', read_name, read.mapping_quality])
                else:
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[2]
                    candidate.append([ref_chr, start, 'rev', ref_chr_next, end, 'rev', 'BND', read_name, read.mapping_quality])
            else:
                if alignment_current[-1] == '+':
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[3]
                    candidate.append([ref_chr, start, 'fwd', ref_chr_next, end, 'rev', 'BND', read_name, read.mapping_quality])
                else:
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[2]
                    candidate.append([ref_chr, start, 'rev', ref_chr_next, end, 'fwd', 'BND', read_name, read.mapping_quality])


# ============================================================================
# CIGAR Processing for Feature Generation
# ============================================================================

def mergecigar_del(infor: List) -> List:
    """Merge deletion events from CIGAR"""
    data = []
    i = 0
    while i < len(infor):
        count = 0
        for j in range(i + 1, len(infor)):
            if abs(infor[j][0] - infor[i][1]) <= 150:
                count += 1
                infor[i][1] = infor[j][1]
        length = abs(infor[i][0] - infor[i][1])
        data.append([infor[i][0], infor[i][1], length])
        i += count + 1
    return data


def mergecigar_ins(infor: List) -> List:
    """Merge insertion events from CIGAR"""
    data = []
    i = 0
    while i < len(infor):
        count = 0
        length = infor[i][2]
        for j in range(i + 1, len(infor)):
            if abs(infor[j][1] - infor[i][1]) <= 150:
                count += 1
                infor[i][1] = infor[j][0]
                length += infor[j][2]
        data.append([infor[i][0], infor[i][0] + 1, length])
        i += count + 1
    return data


def cigarread(read) -> Tuple[List, List, List, List]:
    """Process CIGAR string to extract indels and clips"""
    candidate_ins = []
    candidate_del = []
    loci_clip_sm = []
    loci_clip_ms = []
    
    aligned_length = read.reference_length or 0
    read_name = read.query_name
    chr_name = read.reference_name
    
    cigar_del = []
    cigar_ins = []
    sta = read.reference_start
    
    for ci in read.cigartuples:
        if ci[0] in [0, 7, 8]:
            sta += ci[1]
        elif ci[0] == 2:  # Deletion
            if ci[1] >= 40:
                cigar_del.append([sta, sta + ci[1], ci[1]])
            sta += ci[1]
        elif ci[0] == 1:  # Insertion
            if ci[1] >= 40:
                cigar_ins.append([sta, sta, ci[1]])
    
    if cigar_del:
        cigar_del.sort(key=lambda x: x[0])
        cigar_del = mergecigar_del(cigar_del)
        for del_cigar in cigar_del:
            candidate_del.append([del_cigar[0], del_cigar[2], read_name, 'None', 'DEL', chr_name, read.mapping_quality])
    
    if cigar_ins:
        cigar_ins.sort(key=lambda x: x[0])
        cigar_ins = mergecigar_ins(cigar_ins)
        for ins_ci in cigar_ins:
            candidate_ins.append([ins_ci[0], ins_ci[2], read_name, 'None', 'INS', chr_name, read.mapping_quality])
    
    # Process clips
    if read.cigartuples[-1][0] in [4, 5]:
        if read.is_reverse:
            loci_clip_ms.append(read.reference_end)
        else:
            loci_clip_ms.append(read.reference_start)
    
    if read.cigartuples[0][0] in [4, 5]:
        if read.is_reverse:
            loci_clip_sm.append(read.reference_end)
        else:
            loci_clip_sm.append(read.reference_start)
    
    return candidate_del, candidate_ins, loci_clip_sm, loci_clip_ms


def loci_read_count(read) -> np.ndarray:
    """Get reference positions covered by read"""
    return np.array(read.get_reference_positions())


# ============================================================================
# Signal Analysis Functions
# ============================================================================

def analysis_cigar_indels(del_cigar: List, ins_cigar: List) -> Tuple[List, List]:
    """Analyze CIGAR indels and expand to position lists"""
    sv_sig_del = []
    sv_sig_ins = []
    
    for cigar in del_cigar:
        data = np.arange(cigar[0], cigar[0] + cigar[1]).tolist()
        sv_sig_del.extend(data)
    
    for cigar in ins_cigar:
        data = np.arange(cigar[0], cigar[0] + 1).tolist()
        sv_sig_ins.extend(data)

    return sv_sig_del, sv_sig_ins


def _split_candidate_type(sv_can: List) -> str:
    if len(sv_can) >= 7 and sv_can[6] == 'BND':
        return 'BND'
    if len(sv_can) >= 5:
        return sv_can[4]
    raise ValueError(f"Unsupported split candidate format: {sv_can}")


def analysis_splitread_data(split_read_candidate: List) -> Tuple[List, List, List, List, List]:
    """Analyze split read data and expand to position lists"""
    sv_sig_del = []
    sv_sig_ins = []
    sv_sig_inv = []
    sv_sig_dup = []
    sv_sig_bnd = []
    
    for sv_can in split_read_candidate:
        sv_type = _split_candidate_type(sv_can)
        
        if sv_type == 'DEL' and sv_can[1] <= 1000000:
            data = np.arange(sv_can[0], sv_can[1] + sv_can[0]).tolist()
            sv_sig_del.extend(data)
        elif sv_type == 'INS':
            data = np.arange(sv_can[0], sv_can[0] + 1).tolist()
            sv_sig_ins.extend(data)
        elif sv_type == 'INV' and sv_can[1] <= 1000000:
            data = np.arange(sv_can[0], sv_can[1] + sv_can[0]).tolist()
            sv_sig_inv.extend(data)
        elif sv_type == 'DUP' and sv_can[1] <= 1000000:
            data = np.arange(sv_can[0], sv_can[1] + sv_can[0]).tolist()
            sv_sig_dup.extend(data)
        elif sv_type == 'BND':
            data = np.arange(sv_can[1], sv_can[1] + 1).tolist()
            data1 = np.arange(sv_can[4], sv_can[4] + 1).tolist()
            sv_sig_bnd.extend(data)
            sv_sig_bnd.extend(data1)
    
    return sv_sig_del, sv_sig_ins, sv_sig_inv, sv_sig_dup, sv_sig_bnd


def _build_legacy_signal_matrix(
    del_cigar_all_rev, ins_cigar_all_rev, del_split_all_rev, ins_split_all_rev,
    inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev, loci_read_all_rev,
    clip_sm_all_rev, clip_ms_all_rev, del_cigar_all_fwd, ins_cigar_all_fwd,
    del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd,
    bnd_split_all_fwd, loci_read_all_fwd, clip_sm_all_fwd, clip_ms_all_fwd,
    start: int, end: int
) -> np.ndarray:
    """Compute position-wise signal counts for all features"""
    offset = int(end - start)
    
    def process_signal(signal_list: List) -> np.ndarray:
        """Convert signal list to bincount array"""
        if len(signal_list) != 0:
            arr = np.array(signal_list) - start
            return np.bincount(arr[arr >= 0], minlength=offset + 1)[0:offset].reshape(-1, 1)
        return np.zeros([offset, 1])
    
    signals = [
        process_signal(del_cigar_all_rev),
        process_signal(ins_cigar_all_rev),
        process_signal(del_split_all_rev),
        process_signal(ins_split_all_rev),
        process_signal(inv_split_all_rev),
        process_signal(dup_split_all_rev),
        process_signal(bnd_split_all_rev),
        process_signal(loci_read_all_rev),
        process_signal(clip_sm_all_rev),
        process_signal(clip_ms_all_rev),
        process_signal(del_cigar_all_fwd),
        process_signal(ins_cigar_all_fwd),
        process_signal(del_split_all_fwd),
        process_signal(ins_split_all_fwd),
        process_signal(inv_split_all_fwd),
        process_signal(dup_split_all_fwd),
        process_signal(bnd_split_all_fwd),
        process_signal(loci_read_all_fwd),
        process_signal(clip_sm_all_fwd),
        process_signal(clip_ms_all_fwd),
    ]
    return np.concatenate(signals, axis=1)


def compute_loci_legacy(
    del_cigar_all_rev, ins_cigar_all_rev, del_split_all_rev, ins_split_all_rev,
    inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev, loci_read_all_rev,
    clip_sm_all_rev, clip_ms_all_rev, del_cigar_all_fwd, ins_cigar_all_fwd,
    del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd,
    bnd_split_all_fwd, loci_read_all_fwd, clip_sm_all_fwd, clip_ms_all_fwd,
    start: int, end: int
) -> np.ndarray:
    """Compute the legacy 20-channel feature matrix."""
    infor = _build_legacy_signal_matrix(
        del_cigar_all_rev, ins_cigar_all_rev, del_split_all_rev, ins_split_all_rev,
        inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev, loci_read_all_rev,
        clip_sm_all_rev, clip_ms_all_rev, del_cigar_all_fwd, ins_cigar_all_fwd,
        del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd,
        bnd_split_all_fwd, loci_read_all_fwd, clip_sm_all_fwd, clip_ms_all_fwd,
        start, end,
    )
    return normalize_features(infor)


def compute_loci(
    del_cigar_all_rev, ins_cigar_all_rev, del_split_all_rev, ins_split_all_rev,
    inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev, loci_read_all_rev,
    clip_sm_all_rev, clip_ms_all_rev, del_cigar_all_fwd, ins_cigar_all_fwd,
    del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd,
    bnd_split_all_fwd, loci_read_all_fwd, clip_sm_all_fwd, clip_ms_all_fwd,
    start: int, end: int, feature_version: str = DEFAULT_FEATURE_VERSION,
    split_candidates_rev: Optional[List] = None, split_candidates_fwd: Optional[List] = None,
    del_cigar_candidates_rev: Optional[List] = None, ins_cigar_candidates_rev: Optional[List] = None,
    del_cigar_candidates_fwd: Optional[List] = None, ins_cigar_candidates_fwd: Optional[List] = None,
) -> np.ndarray:
    get_feature_spec(feature_version)
    return compute_loci_legacy(
        del_cigar_all_rev, ins_cigar_all_rev, del_split_all_rev, ins_split_all_rev,
        inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev, loci_read_all_rev,
        clip_sm_all_rev, clip_ms_all_rev, del_cigar_all_fwd, ins_cigar_all_fwd,
        del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd,
        bnd_split_all_fwd, loci_read_all_fwd, clip_sm_all_fwd, clip_ms_all_fwd,
        start,
        end,
    )


def normalize_features(a: np.ndarray) -> np.ndarray:
    """Normalize features using z-score normalization"""
    a = a.astype(np.float32)
    a -= a.mean(axis=0)
    a /= (np.sqrt(a.var(axis=0)) + 1e-10)
    return a


def _normalize_contig_key(contig: str) -> str:
    contig = str(contig)
    return contig[3:] if contig.startswith('chr') else contig


def load_bed_intervals_by_chrom(bed_file: Optional[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if not bed_file:
        return {}

    bed_df = pd.read_csv(
        bed_file,
        sep='\t',
        header=None,
        usecols=[0, 1, 2],
        names=['chrom', 'start', 'end'],
        engine='python',
    )
    bed_df['chrom'] = bed_df['chrom'].astype(str).map(_normalize_contig_key)

    intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for chrom, group in bed_df.groupby('chrom'):
        starts = np.sort(group['start'].to_numpy(dtype=np.int64))
        ends = np.sort(group['end'].to_numpy(dtype=np.int64))
        intervals[str(chrom)] = (starts, ends)
    return intervals


def _region_overlaps_bed(start: int, end: int, interval_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> bool:
    if interval_data is None:
        return True
    starts, ends = interval_data
    if len(starts) == 0:
        return False
    start_count = np.searchsorted(starts, end, side='left')
    end_count = np.searchsorted(ends, start, side='right')
    return bool(start_count > end_count)


def _window_keep_mask(
    index: np.ndarray,
    window_size: int,
    interval_data: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if interval_data is None:
        return np.ones(len(index), dtype=bool)
    starts, ends = interval_data
    if len(starts) == 0:
        return np.zeros(len(index), dtype=bool)
    w_start = index.astype(np.int64, copy=False)
    w_end = w_start + int(window_size)
    start_count = np.searchsorted(starts, w_end, side='left')
    end_count = np.searchsorted(ends, w_start, side='right')
    return start_count > end_count


def _generate_subregion_size(window_size: int, start: int, end: int) -> int:
    raw_value = os.environ.get(GENERATE_SUBREGION_SIZE_ENV, "").strip()
    if not raw_value:
        return 0

    try:
        size = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected positive integer",
            GENERATE_SUBREGION_SIZE_ENV,
            raw_value,
        )
        return 0

    if size <= 0:
        return 0

    size = max(int(window_size), size)
    remainder = size % int(window_size)
    if remainder != 0:
        size -= remainder
        logger.warning(
            "Rounded %s down to %d to match window_size=%d",
            GENERATE_SUBREGION_SIZE_ENV,
            size,
            window_size,
        )
    if size <= 0 or (end - start) <= size:
        return 0
    return size


def _create_region_data(
    bamfile_long_path: str,
    chr_name_long: str,
    start: int,
    end: int,
    feature_version: str,
    bam_threads: int,
    worker_bam,
    window_size: int,
) -> np.ndarray:
    subregion_size = _generate_subregion_size(window_size, start, end)
    if subregion_size <= 0:
        return create_data(
            bamfile_long_path,
            chr_name_long,
            start,
            end,
            feature_version=feature_version,
            bam_threads=bam_threads,
            bamfile=worker_bam,
        )

    logger.info(
        "Processing %s:%d-%d in subregions of %d bp",
        chr_name_long,
        start,
        end,
        subregion_size,
    )

    parts = []
    for sub_start in range(start, end, subregion_size):
        sub_end = min(end, sub_start + subregion_size)
        logger.info(
            "Processing subregion %s:%d-%d",
            chr_name_long,
            sub_start,
            sub_end,
        )
        part = create_data(
            bamfile_long_path,
            chr_name_long,
            sub_start,
            sub_end,
            feature_version=feature_version,
            bam_threads=bam_threads,
            bamfile=worker_bam,
        )
        parts.append(part)

    if not parts:
        return np.empty((0, get_feature_spec(feature_version)['feature_dim']), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


# ============================================================================
# Main Data Generation Functions
# ============================================================================

def create_data(
    bamfile_long_path: str,
    chr_name: str,
    start: int,
    end: int,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    bam_threads: int = 2,
    bamfile=None,
) -> np.ndarray:
    """Create feature data for a genomic region"""
    get_feature_spec(feature_version)
    owns_bam_handle = bamfile is None
    if owns_bam_handle:
        bamfile = pysam.AlignmentFile(bamfile_long_path, 'rb', threads=max(1, int(bam_threads)))
    
    time_start = time.time()
    
    # Initialize signal containers for forward and reverse strands
    loci_read_all_rev = []
    loci_read_all_fwd = []
    del_cigar_all_rev = []
    ins_cigar_all_rev = []
    clip_sm_all_rev = []
    clip_sm_all_fwd = []
    clip_ms_all_rev = []
    clip_ms_all_fwd = []
    del_cigar_all_fwd = []
    ins_cigar_all_fwd = []
    split_read_candidate_fwd = []
    split_read_candidate_rev = []
    
    try:
        for read in bamfile.fetch(chr_name, start, end):
            if read.is_unmapped or read.is_duplicate or read.is_secondary or read.is_supplementary:
                continue
            
            read_flag = detect_flag(read.flag)
            
            if read_flag % 2 == 0:  # Reverse strand
                del_cigar, ins_cigar, clip_sm, clip_ms = cigarread(read)
                del_cigar_all_rev.extend(del_cigar)
                ins_cigar_all_rev.extend(ins_cigar)
                clip_sm_all_rev.extend(clip_sm)
                clip_ms_all_rev.extend(clip_ms)
                
                if read.has_tag('SA'):
                    splitread = splitreadlist(read)
                    analyze_read_segments(read, feature_read_segment(splitread), split_read_candidate_rev)
                
                loci_read = loci_read_count(read)
                loci_read_all_rev.extend(loci_read)
            else:  # Forward strand
                del_cigar, ins_cigar, clip_sm, clip_ms = cigarread(read)
                del_cigar_all_fwd.extend(del_cigar)
                ins_cigar_all_fwd.extend(ins_cigar)
                clip_sm_all_fwd.extend(clip_sm)
                clip_ms_all_fwd.extend(clip_ms)
                
                if read.has_tag('SA'):
                    splitread = splitreadlist(read)
                    analyze_read_segments(read, feature_read_segment(splitread), split_read_candidate_fwd)
                
                loci_read = loci_read_count(read)
                loci_read_all_fwd.extend(loci_read)
        
        # Analyze signals
        del_cigar_all_rev_legacy, ins_cigar_all_rev_legacy = analysis_cigar_indels(del_cigar_all_rev, ins_cigar_all_rev)
        del_split_all_rev, ins_split_all_rev, inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev = (
            analysis_splitread_data(split_read_candidate_rev)
        )
        
        del_cigar_all_fwd_legacy, ins_cigar_all_fwd_legacy = analysis_cigar_indels(del_cigar_all_fwd, ins_cigar_all_fwd)
        del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd, bnd_split_all_fwd = (
            analysis_splitread_data(split_read_candidate_fwd)
        )
        
        # Compute position-wise features
        all_data = compute_loci(
            del_cigar_all_rev_legacy, ins_cigar_all_rev_legacy, del_split_all_rev, ins_split_all_rev,
            inv_split_all_rev, dup_split_all_rev, bnd_split_all_rev, loci_read_all_rev,
            clip_sm_all_rev, clip_ms_all_rev, del_cigar_all_fwd_legacy, ins_cigar_all_fwd_legacy,
            del_split_all_fwd, ins_split_all_fwd, inv_split_all_fwd, dup_split_all_fwd,
            bnd_split_all_fwd, loci_read_all_fwd, clip_sm_all_fwd, clip_ms_all_fwd,
            start, end,
            feature_version=feature_version,
            split_candidates_rev=split_read_candidate_rev,
            split_candidates_fwd=split_read_candidate_fwd,
            del_cigar_candidates_rev=del_cigar_all_rev,
            ins_cigar_candidates_rev=ins_cigar_all_rev,
            del_cigar_candidates_fwd=del_cigar_all_fwd,
            ins_cigar_candidates_fwd=ins_cigar_all_fwd,
        )
        
        gc.collect()
        logger.debug(f"Processed {chr_name}:{start}-{end} in {time.time() - time_start:.2f}s")
        return all_data
    finally:
        if owns_bam_handle and bamfile is not None:
            bamfile.close()


def _init_generate_worker(bamfile_long_path: str, bam_threads: int):
    """Open one BAM handle per worker and reuse it across regions."""
    global _GENERATE_WORKER_BAM, _GENERATE_WORKER_BAM_PATH
    _GENERATE_WORKER_BAM_PATH = str(bamfile_long_path)
    _GENERATE_WORKER_BAM = pysam.AlignmentFile(
        bamfile_long_path,
        'rb',
        threads=max(1, int(bam_threads)),
    )


def process_region(
    bamfile_long_path: str,
    chr_name_long: str,
    start: int,
    end: int,
    outputpath: str,
    window_size: int,
    feature_version: str,
    bed_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    feature_dtype: str = 'float16',
    storage_format: str = 'npy',
    overwrite: bool = False,
    bam_threads: int = 2,
):
    """Process a genomic region and save features"""
    time_s = time.time()
    output_store = feature_store_path(
        outputpath,
        chr_name_long,
        start,
        end,
        storage_format=storage_format,
    )

    if not overwrite and feature_store_complete(output_store):
        logger.info(f"Skipping existing region store {output_store}")
        return
    if bed_intervals is not None and not _region_overlaps_bed(start, end, bed_intervals):
        logger.info(f"Skipping {chr_name_long}:{start}-{end} (no BED overlap)")
        return
    
    logger.info(f"Processing region {chr_name_long}:{start}-{end}")
    worker_bam = None
    if _GENERATE_WORKER_BAM is not None and _GENERATE_WORKER_BAM_PATH == str(bamfile_long_path):
        worker_bam = _GENERATE_WORKER_BAM
    
    all_data = _create_region_data(
        bamfile_long_path,
        chr_name_long,
        start,
        end,
        feature_version,
        bam_threads,
        worker_bam,
        window_size,
    )
    usable_rows = (all_data.shape[0] // window_size) * window_size
    if usable_rows != all_data.shape[0]:
        all_data = all_data[:usable_rows]

    feature_dim = int(all_data.shape[1]) if all_data.ndim == 2 else get_feature_spec(feature_version)['feature_dim']
    all_data = all_data.reshape(-1, window_size, feature_dim)
    index = np.arange(start, start + usable_rows, window_size, dtype=np.int32)

    keep_mask = _window_keep_mask(index, window_size, bed_intervals)
    all_data = all_data[keep_mask]
    index = index[keep_mask]

    storage_dtype = np.dtype(feature_dtype)
    all_data = all_data.astype(storage_dtype, copy=False)
    canonical_contig = (
        chr_name_long
        if str(chr_name_long).startswith('chr')
        else f"chr{chr_name_long}"
    )

    save_npy_feature_store(
        output_store,
        data=all_data,
        index=index.astype(np.int32, copy=False),
        overwrite=True,
    )
    
    gc.collect()
    logger.info(
        "Completed %s:%d-%d in %.2fs (%d windows kept)",
        chr_name_long,
        start,
        end,
        time.time() - time_s,
        int(len(index)),
    )


def _run_region_tasks_serial(
    region_tasks: List[Tuple],
    bamfile_long_path: str,
    bam_threads: int,
):
    """Fallback execution path when multiprocessing is unavailable."""
    global _GENERATE_WORKER_BAM, _GENERATE_WORKER_BAM_PATH

    _init_generate_worker(bamfile_long_path, bam_threads)
    try:
        for task in region_tasks:
            process_region(*task)
    finally:
        if _GENERATE_WORKER_BAM is not None:
            _GENERATE_WORKER_BAM.close()
        _GENERATE_WORKER_BAM = None
        _GENERATE_WORKER_BAM_PATH = None


def create_data_long(
    bamfile_long_path: str,
    outputpath: str,
    contig: List,
    window_size: int = 2000,
    threadss: int = 10,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    bed_file: Optional[str] = None,
    feature_dtype: str = 'float16',
    storage_format: str = 'npy',
    overwrite: bool = False,
    bam_threads: int = 2,
):
    """
    Create feature data for all contigs from a BAM file
    
    Args:
        bamfile_long_path: Path to BAM file
        outputpath: Output directory for feature files
        contig: List of contigs to process (empty for all)
        window_size: Window size for features (default: 2000)
        threadss: Number of threads
    """
    time_st = time.time()
    threadss = max(1, int(threadss))
    bam_threads = max(1, int(bam_threads))
    write_feature_config(
        outputpath,
        feature_version=feature_version,
        window_size=window_size,
        storage_format=storage_format,
        storage_dtype=feature_dtype,
    )
    bed_intervals_by_chr = load_bed_intervals_by_chrom(bed_file)
    
    bamfile_long = pysam.AlignmentFile(bamfile_long_path, 'rb', threads=1)
    contig2length = {}
    
    if len(contig) == 0:
        contig = []
        for count in range(len(bamfile_long.get_index_statistics())):
            contig.append(bamfile_long.get_index_statistics()[count].contig)
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    else:
        contig = np.array(contig).astype(str)
    
    for count in range(len(bamfile_long.get_index_statistics())):
        contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    
    bamfile_long.close()
    
    region_tasks = []
    for ww in contig:
        chr_name_long = str(ww)  # Convert numpy.str_ to Python str
        chr_length = contig2length[str(ww)]  # Use string key for lookup
        interval_data = bed_intervals_by_chr.get(_normalize_contig_key(chr_name_long))
        ider = math.ceil(chr_length / REGION_SIZE)
        
        logger.info(f"Processing chromosome {chr_name_long} ({ider} segments)")
        time_q = time.time()
        scheduled = 0

        for current_start in range(0, chr_length, REGION_SIZE):
            current_end = current_start + REGION_SIZE

            output_store = feature_store_path(
                outputpath,
                chr_name_long,
                current_start,
                current_end,
                storage_format=storage_format,
            )
            if not overwrite and feature_store_complete(output_store):
                logger.info(f"Skipping existing region store {output_store}")
                continue
            if interval_data is not None and not _region_overlaps_bed(current_start, current_end, interval_data):
                logger.info(f"Skipping {chr_name_long}:{current_start}-{current_end} (no BED overlap)")
                continue
            region_tasks.append((
                bamfile_long_path,
                chr_name_long,
                current_start,
                current_end,
                outputpath,
                window_size,
                feature_version,
                interval_data,
                feature_dtype,
                storage_format,
                overwrite,
                bam_threads,
            ))
            scheduled += 1
        logger.info(
            "Chromosome %s scheduled in %.2fs (%d region tasks)",
            chr_name_long,
            time.time() - time_q,
            scheduled,
        )

    if not region_tasks:
        logger.info("No regions require generation")
        logger.info(f"Total processing time: {time.time() - time_st:.2f}s")
        return

    if threadss <= 1:
        logger.info("Dispatching %d regions serially (1 worker)", len(region_tasks))
        _run_region_tasks_serial(region_tasks, bamfile_long_path, bam_threads)
        logger.info(f"Total processing time: {time.time() - time_st:.2f}s")
        return

    logger.info("Dispatching %d regions across a fixed worker pool (%d workers)", len(region_tasks), threadss)
    try:
        with Pool(
            processes=threadss,
            initializer=_init_generate_worker,
            initargs=(bamfile_long_path, bam_threads),
        ) as pool:
            pool.starmap(process_region, region_tasks, chunksize=1)
    except PermissionError:
        logger.warning(
            "Multiprocessing pool setup failed; falling back to serial region generation"
        )
        _run_region_tasks_serial(region_tasks, bamfile_long_path, bam_threads)
    
    logger.info(f"Total processing time: {time.time() - time_st:.2f}s")


# ============================================================================
# Utility Functions
# ============================================================================

def average_read_coverage(bamfile, chr_name: str, lengthh: int) -> float:
    """Calculate average read coverage for a chromosome"""
    all_length = lengthh
    chr_align_length = 0
    
    for read in bamfile.fetch(chr_name):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        chr_align_length += read.query_length
    
    depth_coverage = np.ceil(int(chr_align_length) / int(all_length))
    return depth_coverage


def labeldata(vcfpath: str, contig: str, start: int, end: int, 
              window_size: int, index: np.ndarray) -> np.ndarray:
    """Generate labels from VCF file"""
    goldl = []
    
    if 'chr' in contig:
        contig = contig[3:]
    
    for rec in pysam.VariantFile(vcfpath).fetch():
        if rec.contig != contig:
            continue
        if rec.info['SVTYPE'] in ['INS', 'DEL', 'INV', 'DUP']:
            if rec.stop is None: # Add check for None
                continue
            goldl.append([rec.start, rec.stop, rec.stop - rec.start, 1])
    
    if not goldl:  # If no variants were found
        return np.zeros(len(index), dtype='float32')

    goldl = pd.DataFrame(goldl, columns=['start', 'stop', 'len', 'val']).sort_values(['start', 'stop']).values.astype('float64')
    y = []
    for rec in index:
        if ((goldl[:, 1:2] > rec) & (goldl[:, :1] < (rec + window_size))).sum() != 0:
            y.append((((goldl[:, 1:2] >= rec) & (goldl[:, :1] <= (rec + window_size))) * goldl[:, 3:]).sum())
        else:
            y.append(0)
    
    return (np.array(y) > 0).astype('float32')


def labelbed(bed_file: str, contig: str, start: int, end: int,
             window_size: int, index: np.ndarray) -> np.ndarray:
    """Generate labels from BED file"""
    bnd_data = pd.read_csv(bed_file, sep='\t', header=None).values.tolist()
    bnd_data = sorted(bnd_data, key=lambda x: x[-1])
    
    chr_values, start_values, svlen_values = [], [], []
    svtype_values = []
    
    for i in range(len(bnd_data)):
        if bnd_data[i][4] != 'TRA':
            chr_values.append(bnd_data[i][0])
            start_values.append(bnd_data[i][1])
            svlen_values.append(bnd_data[i][3] - bnd_data[i][1])
        else:
            chr_values.append(bnd_data[i][0])
            start_values.append(bnd_data[i][1])
            svlen_values.append(f"{bnd_data[i][2]}:{bnd_data[i][3]}")
        svtype_values.append(bnd_data[i][4])
    
    result_dict = {}
    for i in range(len(chr_values)):
        if chr_values[i] in result_dict:
            result_dict[chr_values[i]].append([start_values[i], svlen_values[i], svtype_values[i]])
        else:
            result_dict[chr_values[i]] = [[start_values[i], svlen_values[i], svtype_values[i]]]
    
    goldl = []
    if 'chr' in contig:
        contig = contig[3:]
    
    for rec in result_dict.get(contig, []):
        if rec[2] in ['DEL', 'INV', 'DUP']:
            goldl.append([rec[0], rec[0] + rec[1], rec[1], 1])
        elif rec[2] in ['TRA', 'INS']:
            goldl.append([rec[0], rec[0] + 1, 1, 1])
    
    if goldl:
        goldl = pd.DataFrame(goldl).sort_values([0, 1]).values.astype('float64')
    else:
        goldl = np.empty((0, 4), dtype='float64')

    y = []
    for rec in index:
        if goldl.shape[0] == 0:
            y.append(0)
            continue

        mask = (goldl[:, 1] >= rec) & (goldl[:, 0] <= (rec + window_size))
        if mask.any():
            y.append(float(goldl[mask, -1].sum()))
        else:
            y.append(0)
    
    return (np.array(y) > 0).astype('float32')


if __name__ == '__main__':
    print("CMSV Data Generation Module (PyTorch Version)")
    print("Usage: create_data_long(bamfile_path, output_path, contigs, window_size, threads)")
