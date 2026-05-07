"""
CMSV Detection Module (PyTorch Version)
Contains functions for SV detection, clustering, and VCF generation
"""

import pysam
import numpy as np
import math
import time
import os
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
from multiprocessing import Pool
from math import log10
from statistics import mean

import torch
import torch.nn as nn

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binom

from cmsv_features import resolve_feature_config
from cmsv_model import create_model
from cmsv_platform import resolve_platform_id, resolve_platform_name
from cmsv_storage import load_feature_data, load_feature_index, resolve_feature_store_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cap implausibly large DUP/INV events by default.
# Set to 0 to disable the cap.
DEFAULT_MAX_DUP_INV_LEN = 2_000_000
DEFAULT_DECISION_THRESHOLD = 0.5
REGION_SIZE = 10_000_000
INS_EPS_BANDS_NOISY = [(300, 150), (1000, 220), (5000, 300), (10**12, 420)]
INS_EPS_BANDS_CLEAN = [(300, 220), (1000, 300), (5000, 420), (10**12, 600)]
INV_EPS_BANDS_NOISY = [(3000, 280), (10000, 420), (10**12, 650)]
INV_EPS_BANDS_CLEAN = [(3000, 360), (10000, 520), (10**12, 800)]
MIN_SPLIT_MAPQ = 20
MIN_SPLIT_SEGMENT_SPAN = 200
LONG_INS_MIN_SPLIT_MAPQ = 10
LONG_INS_MIN_SEGMENT_SPAN = 80
SHORT_INV_LEN = 5_000
SHORT_INV_MIN_SEGMENT_SPAN = 500
SHORT_INV_MAX_READ_OVERLAP = 200
SHORT_INV_MAX_READ_GAP = 500
LONG_INS_LEN = 1_000


# ============================================================================
# CIGAR and Split-read Analysis Functions
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


# Signal detection flags
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


def _segment_span(segment: List) -> int:
    """Aligned span on the read for one split-read segment."""
    return max(0, int(segment[1]) - int(segment[0]))


def _segment_mapq(segment: List) -> int:
    """Supplementary MAPQ stored alongside split-read segments."""
    if len(segment) >= 7:
        return int(segment[4])
    return MIN_SPLIT_MAPQ


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
        sv_list.append([qry_start, qry_end, qry_reference_start, qry_reference_end, read.mapq, read.reference_name, strand_])
        
        rawsalist = read.get_tag('SA').split(';')
        for sa in rawsalist[:-1]:
            sainfo = sa.split(',')
            tmpcontig, tmprefstart, strand, cigar, sup_mapq = (
                sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3], int(sainfo[4])
            )
            if sup_mapq < LONG_INS_MIN_SPLIT_MAPQ:
                continue
            refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
            if min(readend_2 - readstart_2, refend_2 - refstart_2) < LONG_INS_MIN_SEGMENT_SPAN:
                continue

            if strand == '-' and sup_mapq >= 0:
                readstart = read.query_length - readend_2
                readend = read.query_length - readstart_2
                sv_list.append([readstart, readend, refstart_2, refend_2, sup_mapq, tmpcontig, strand])
            elif strand == '+' and sup_mapq >= 0:
                sv_list.append([readstart_2, readend_2, refstart_2, refend_2, sup_mapq, tmpcontig, strand])

    return sv_list


def feature_record(alignment_current: List, alignment_next: List, ins_tra_flag: bool = False) -> Tuple:
    """Calculate features between two aligned segments"""
    distance_on_read = alignment_next[0] - alignment_current[1]
    
    if alignment_current[-1] == '+':
        distance_on_reference = alignment_next[2] - alignment_current[3]
        if alignment_next[-1] == '-':  # INV:+-
            if alignment_current[3] > alignment_next[3]:
                distance_on_reference = alignment_next[3] - alignment_current[2]
            else:
                distance_on_reference = alignment_current[3] - alignment_next[2]
    else:
        distance_on_reference = alignment_current[2] - alignment_next[3]
        if alignment_next[-1] == '+':  # INV:-+
            if alignment_current[3] > alignment_next[3]:
                distance_on_reference = alignment_next[3] - alignment_current[2]
            else:
                distance_on_reference = alignment_current[3] - alignment_next[2]
    
    deviation = distance_on_read - distance_on_reference
    chr_ = 1 if alignment_current[-2] == alignment_next[-2] else 0
    orientation = 1 if alignment_current[-1] == alignment_next[-1] else 0
    
    return (alignment_current, alignment_next, chr_, orientation, distance_on_read, 
            distance_on_reference, deviation, ins_tra_flag)


def feature_read_segment(svlist: List) -> List:
    """Extract features from read segments"""
    sg_list = []
    sorted_alignment_list = sorted(svlist, key=lambda aln: (aln[0], aln[1]))
    
    for index in range(len(sorted_alignment_list) - 1):
        sg_list.append(feature_record(sorted_alignment_list[index], sorted_alignment_list[index + 1]))
    
    if len(svlist) >= 3 and sorted_alignment_list[0][-2] != sorted_alignment_list[1][-2]:
        sg_list.append(feature_record(sorted_alignment_list[0], sorted_alignment_list[-1], ins_tra_flag=True))
    
    return sg_list


def _valid_short_inv_signature(alignment_current: List, alignment_next: List, start: int, end: int) -> bool:
    """Filter noisy short INV signatures before they become candidates."""
    inv_len = int(end) - int(start)
    if inv_len >= SHORT_INV_LEN:
        return True

    current_span = _segment_span(alignment_current)
    next_span = _segment_span(alignment_next)
    if min(current_span, next_span) < SHORT_INV_MIN_SEGMENT_SPAN:
        return False

    read_gap = int(alignment_next[0]) - int(alignment_current[1])
    if read_gap < -SHORT_INV_MAX_READ_OVERLAP:
        return False
    if read_gap > SHORT_INV_MAX_READ_GAP:
        return False
    return True


def _valid_ins_signature(alignment_current: List, alignment_next: List, svlen: int) -> bool:
    """Allow lower-MAPQ supplementary evidence only for long INS rescue."""
    min_mapq = min(_segment_mapq(alignment_current), _segment_mapq(alignment_next))
    min_span = min(_segment_span(alignment_current), _segment_span(alignment_next))
    if int(svlen) >= LONG_INS_LEN:
        return min_mapq >= LONG_INS_MIN_SPLIT_MAPQ and min_span >= LONG_INS_MIN_SEGMENT_SPAN
    return min_mapq >= MIN_SPLIT_MAPQ and min_span >= MIN_SPLIT_SEGMENT_SPAN


def analyze_read_segments(
    read,
    segment_data: List,
    candidate: List,
    startt: int,
    endd: int,
    alignscore: float,
    max_dup_inv_len: int = DEFAULT_MAX_DUP_INV_LEN,
):
    """Analyze read segments to identify SV candidates"""
    min_sv_size = 40
    segment_overlap_tolerance = 5
    read_name = read.query_name
    max_dup_inv_len = max(0, int(max_dup_inv_len))
    
    for sv_sig in segment_data:
        alignment_current = sv_sig[0]
        alignment_next = sv_sig[1]
        ref_chr = alignment_current[-2]
        chr_, orientation, distance_on_read, distance_on_reference, deviation, long_ins = sv_sig[2:]
        
        if chr_ == 1:  # Same chromosome
            if orientation == 1:  # Same orientation
                if distance_on_reference >= -min_sv_size or long_ins:  # INS or DEL
                    if deviation > 0:  # INS
                        if alignment_current[-1] == '+':
                            start = ((alignment_current[3] + alignment_next[2]) // 2 
                                    if not long_ins else min(alignment_current[3], alignment_next[2]))
                        else:
                            start = ((alignment_current[2] + alignment_next[3]) // 2 
                                    if not long_ins else min(alignment_current[2], alignment_next[3]))
                        end = start + deviation
                        if end - start < min_sv_size:
                            continue
                        if not _valid_ins_signature(alignment_current, alignment_next, end - start):
                            continue
                        if startt <= start <= endd:
                            candidate.append([start, deviation, read_name, 'A', 'split', 'INS', alignscore, ref_chr])
                    
                    elif deviation < 0:  # DEL
                        if min(_segment_mapq(alignment_current), _segment_mapq(alignment_next)) < MIN_SPLIT_MAPQ:
                            continue
                        if alignment_current[-1] == '+':
                            start = alignment_current[3]
                        else:
                            start = alignment_next[3]
                        end = start - deviation
                        if end - start < min_sv_size:
                            continue
                        if startt <= start <= endd:
                            candidate.append([start, -deviation, read_name, 'None', 'split', 'DEL', alignscore, ref_chr])
                else:  # DUP
                    if min(_segment_mapq(alignment_current), _segment_mapq(alignment_next)) < MIN_SPLIT_MAPQ:
                        continue
                    if alignment_current[-1] == '+':
                        start = alignment_next[2]
                        end = alignment_current[3]
                    else:
                        start = alignment_current[2]
                        end = alignment_next[3]
                    deviation = end - start
                    if deviation < min_sv_size:
                        continue
                    if max_dup_inv_len > 0 and deviation > max_dup_inv_len:
                        continue
                    candidate.append([start, deviation, read_name, 'None', 'split', 'DUP', alignscore, ref_chr])
            
            else:  # INV (different orientation)
                if min(_segment_mapq(alignment_current), _segment_mapq(alignment_next)) < MIN_SPLIT_MAPQ:
                    continue
                if alignment_current[-1] == '+':  # +-
                    if alignment_next[2] - alignment_current[3] >= -segment_overlap_tolerance:
                        start = alignment_current[3]
                        end = alignment_next[3]
                        if startt <= start <= endd:
                            deviation = end - start
                            if deviation < min_sv_size:
                                continue
                            if max_dup_inv_len > 0 and deviation > max_dup_inv_len:
                                continue
                            if not _valid_short_inv_signature(alignment_current, alignment_next, start, end):
                                continue
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])
                    elif alignment_current[2] - alignment_next[3] >= -segment_overlap_tolerance:
                        start = alignment_next[3]
                        end = alignment_current[3]
                        if startt <= start <= endd:
                            deviation = end - start
                            if deviation < min_sv_size:
                                continue
                            if max_dup_inv_len > 0 and deviation > max_dup_inv_len:
                                continue
                            if not _valid_short_inv_signature(alignment_current, alignment_next, start, end):
                                continue
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])
                else:  # -+
                    if alignment_next[2] - alignment_current[3] >= -segment_overlap_tolerance:
                        start = alignment_current[2]
                        end = alignment_next[2]
                        if startt <= start <= endd:
                            deviation = end - start
                            if deviation < min_sv_size:
                                continue
                            if max_dup_inv_len > 0 and deviation > max_dup_inv_len:
                                continue
                            if not _valid_short_inv_signature(alignment_current, alignment_next, start, end):
                                continue
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])
                    elif alignment_current[2] - alignment_next[3] >= -segment_overlap_tolerance:
                        start = alignment_next[2]
                        end = alignment_current[2]
                        if startt <= start <= endd:
                            deviation = end - start
                            if deviation < min_sv_size:
                                continue
                            if max_dup_inv_len > 0 and deviation > max_dup_inv_len:
                                continue
                            if not _valid_short_inv_signature(alignment_current, alignment_next, start, end):
                                continue
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])
        
        else:  # Different chromosomes - Translocation
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
                    candidate.append([start, read_name, 'fwd', ref_chr_next, end, 'fwd', 'TRA', alignscore, ref_chr])
                else:
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[2]
                    candidate.append([start, read_name, 'rev', ref_chr_next, end, 'rev', 'TRA', alignscore, ref_chr])
            else:
                if alignment_current[-1] == '+':
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[3]
                    candidate.append([start, read_name, 'fwd', ref_chr_next, end, 'rev', 'TRA', alignscore, ref_chr])
                else:
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[2]
                    candidate.append([start, read_name, 'rev', ref_chr_next, end, 'fwd', 'TRA', alignscore, ref_chr])


# ============================================================================
# CIGAR Processing Functions
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
    if not infor:
        return []

    def merge_window(len_a: int, len_b: int) -> int:
        max_len = max(int(len_a), int(len_b))
        if max_len < 100:
            return 200
        if max_len < 500:
            return 400
        return 600

    rows = sorted(infor, key=lambda x: int(x[0]))
    merged = []
    anchor = int(rows[0][0])
    last_pos = int(rows[0][0])
    total_len = int(rows[0][2])

    for current in rows[1:]:
        current_pos = int(current[0])
        current_len = int(current[2])
        if abs(current_pos - last_pos) <= merge_window(total_len, current_len):
            total_len += current_len
            last_pos = current_pos
            continue
        merged.append([anchor, anchor + 1, total_len])
        anchor = current_pos
        last_pos = current_pos
        total_len = current_len

    merged.append([anchor, anchor + 1, total_len])
    return merged


def _ins_merge_window(len_a: int, len_b: int, source_a: str = '', source_b: str = '') -> int:
    """Read-level merge window for nearby INS evidence."""
    max_len = max(int(len_a), int(len_b))
    if max_len < 100:
        window = 200
    elif max_len < 500:
        window = 400
    elif max_len < 2_000:
        window = 800
    else:
        window = 1_200
    if str(source_a) != str(source_b):
        window += 100
    return window


def merge_read_level_ins_candidates(candidates: List) -> List:
    """Merge nearby split/cigar INS candidates from the same read into one record."""
    if not candidates:
        return []

    ins_rows = [row for row in candidates if len(row) >= 8 and str(row[5]) == 'INS']
    non_ins_rows = [row for row in candidates if not (len(row) >= 8 and str(row[5]) == 'INS')]
    if len(ins_rows) <= 1:
        return candidates

    ins_rows = sorted(ins_rows, key=lambda x: (str(x[7]), int(x[0]), int(x[1])))
    merged_rows = []
    current = [ins_rows[0]]

    def flush(group: List) -> None:
        if not group:
            return
        if len(group) == 1:
            merged_rows.append(group[0])
            return
        starts = [int(item[0]) for item in group]
        lengths = [int(item[1]) for item in group]
        read_name = group[0][2]
        allele = 'A' if any(str(item[3]) == 'A' for item in group) else 'None'
        sources = {str(item[4]) for item in group}
        split_like = [item for item in group if 'split' in str(item[4])]
        long_split_like = [item for item in split_like if int(item[1]) >= LONG_INS_LEN]
        long_group = [item for item in group if int(item[1]) >= LONG_INS_LEN]
        if long_split_like:
            anchor_rows = long_split_like
            merged_length = max(int(item[1]) for item in long_split_like)
            source = 'split_rescue' if len(sources) > 1 else str(long_split_like[0][4])
        elif long_group:
            anchor_rows = long_group
            merged_length = max(int(item[1]) for item in long_group)
            source = 'split+cigar' if len(sources) > 1 else str(group[0][4])
        else:
            anchor_rows = group
            merged_length = int(round(float(np.median(lengths))))
            source = 'split+cigar' if len(sources) > 1 else str(group[0][4])
        alignscore = max(float(item[6]) for item in group)
        ref_chr = group[0][7]
        merged_rows.append([
            int(round(float(np.median([int(item[0]) for item in anchor_rows])))),
            merged_length,
            read_name,
            allele,
            source,
            'INS',
            alignscore,
            ref_chr,
        ])

    for row in ins_rows[1:]:
        prev = current[-1]
        same_chr = str(row[7]) == str(prev[7])
        close = abs(int(row[0]) - int(prev[0])) <= _ins_merge_window(prev[1], row[1], prev[4], row[4])
        if same_chr and close:
            current.append(row)
        else:
            flush(current)
            current = [row]
    flush(current)

    return non_ins_rows + merged_rows


def cigarread(read, candidate: List, start: int, end: int, alignscore: float):
    """Process CIGAR string to extract indel candidates"""
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
            if ci[1] >= 40 and start <= sta + ci[1] and end >= sta:
                cigar_del.append([sta, sta + ci[1], ci[1]])
            sta += ci[1]
        elif ci[0] == 1:  # Insertion
            if ci[1] >= 40 and start <= sta <= end:
                cigar_ins.append([sta, sta, ci[1]])
    
    if cigar_del:
        cigar_del.sort(key=lambda x: x[0])
        cigar_del = mergecigar_del(cigar_del)
        for del_cigar in cigar_del:
            candidate.append([del_cigar[0], del_cigar[2], read_name, 'None', 'cigar', 'DEL', alignscore, chr_name])
    
    if cigar_ins:
        cigar_ins.sort(key=lambda x: x[0])
        cigar_ins = mergecigar_ins(cigar_ins)
        for ins_ci in cigar_ins:
            candidate.append([ins_ci[0], ins_ci[2], read_name, 'None', 'cigar', 'INS', alignscore, chr_name])


# ============================================================================
# Quality Score Functions
# ============================================================================

def alignment_quality_score(mapq: int, error_rate: float, num_of_mismatch: int, mm_rate: float) -> float:
    """Calculate alignment quality score"""
    mapq_weight = 2.0
    error_rate_weight = 5.0
    mismatch_weight = 0.1
    mm_rate_weight = 8.0
    mapq_scale = 0.1
    mapq_threshold = 20
    mismatch_scale = 0.01
    mismatch_threshold = 50
    mm_rate_exponent = 2.0
    
    score = (
        (mapq_weight * math.log(1 + mapq)) / (1 + math.exp(-mapq_scale * (mapq - mapq_threshold))) -
        (error_rate_weight * math.sqrt(error_rate)) -
        (mismatch_weight * num_of_mismatch / (1 + math.exp(-mismatch_scale * (num_of_mismatch - mismatch_threshold)))) -
        (mm_rate_weight * math.pow(mm_rate, mm_rate_exponent))
    )
    return abs(score)


# ============================================================================
# Minimum Support Calculation Functions
# ============================================================================

def calculate_min_support_ins(c_global: float, c_local: float, mu: float, sigma: float, 
                              rho: float, error_rate: float) -> int:
    """Calculate minimum support for insertions"""
    local_deviation_impact = math.tanh((c_local - c_global) / c_global) if c_global > 0 else 0
    
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    else:
        if c_global <= 10:
            mu = 0.6
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    
    return max(1, round(min_support))


def calculate_min_support_del(c_global: float, c_local: float, mu: float, sigma: float,
                              rho: float, error_rate: float) -> int:
    """Calculate minimum support for deletions"""
    if c_global >= 80:
        return 10
    
    local_deviation_impact = math.tanh((c_local - c_global) / c_global) if c_global > 0 else 0
    
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    else:
        if c_global >= 20:
            mu = 0.6
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    
    return max(1, round(min_support))


def calculate_min_support_inv(c_global: float, c_local: float, mu: float, sigma: float,
                              rho: float, error_rate: float) -> int:
    """Calculate minimum support for inversions"""
    if c_global >= 80:
        return 10
    
    local_deviation_impact = math.tanh((c_local - c_global) / c_global) if c_global > 0 else 0
    
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    else:
        if c_global >= 20:
            mu = 0.7
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    
    return max(1, round(min_support))


def calculate_min_support_dup(c_global: float, c_local: float, mu: float, sigma: float,
                              rho: float, error_rate: float) -> int:
    """Calculate minimum support for duplications"""
    if c_global >= 80:
        return 10
    
    local_deviation_impact = math.tanh((c_local - c_global) / c_global) if c_global > 0 else 0
    
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    else:
        if c_global >= 20:
            mu = 0.5
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
    
    return max(1, round(min_support))


# ============================================================================
# Clustering Functions
# ============================================================================

def median(lst: List) -> float:
    """Calculate median of a list"""
    n = len(lst)
    sorted_lst = sorted(lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    return sorted_lst[mid]


def cluster_by_length(lengths: List, threshold_ratio: float = 0.7) -> List:
    """Cluster SVs by length"""
    sorted_lengths = sorted(lengths, key=lambda x: int(x[2]))
    mean_length = int(sorted_lengths[len(sorted_lengths) // 2][2])
    
    clusters = []
    current_cluster = [sorted_lengths[0]]
    
    for length in sorted_lengths[1:]:
        if abs(int(current_cluster[-1][2]) - int(length[2])) <= mean_length * threshold_ratio:
            current_cluster.append(length)
        else:
            clusters.append(current_cluster)
            current_cluster = [length]
    clusters.append(current_cluster)
    
    return clusters


def _trim_ins_length_outliers(rows: List, min_support: int = 2) -> List:
    """Remove obvious INS length outliers before summarizing a cluster."""
    if not rows:
        return []

    trimmed = sorted(rows, key=lambda x: int(x[2]))
    while len(trimmed) > max(2, int(min_support) - 2):
        median_len = int(trimmed[len(trimmed) // 2][2])
        if int(trimmed[-1][2]) > 2 * median_len and int(trimmed[-1][2]) - median_len > 30:
            trimmed = trimmed[:-1]
            continue
        if median_len > 2 * int(trimmed[0][2]) and median_len - int(trimmed[0][2]) > 30:
            trimmed = trimmed[1:]
            continue
        break
    return trimmed


def _parse_eps_band_spec(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse comma-separated max_len:eps band specs."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    bands: List[Tuple[int, int]] = []
    for chunk in text.split(','):
        piece = chunk.strip()
        if not piece:
            continue
        if ':' not in piece:
            raise ValueError(f"Invalid eps band chunk '{piece}', expected max_len:eps")
        max_len_str, eps_str = piece.split(':', 1)
        max_len = int(max_len_str)
        eps = int(eps_str)
        if max_len <= 0 or eps <= 0:
            raise ValueError(f"Invalid eps band '{piece}', values must be > 0")
        bands.append((max_len, eps))
    bands.sort(key=lambda x: x[0])
    return bands or None


def _format_eps_band_spec(bands: Optional[List[Tuple[int, int]]]) -> str:
    if not bands:
        return "disabled"
    return ",".join(f"{max_len}:{eps}" for max_len, eps in bands)


def _length_aware_eps_bands(
    svtype: str,
    error_mean: float,
    override_bands: Optional[List[Tuple[int, int]]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Return per-length DBSCAN radii for selected SV types."""
    if override_bands:
        return override_bands
    noisy = error_mean > 0.1
    if svtype == 'INS':
        return INS_EPS_BANDS_NOISY if noisy else INS_EPS_BANDS_CLEAN
    if svtype == 'INV':
        return INV_EPS_BANDS_NOISY if noisy else INV_EPS_BANDS_CLEAN
    return None


def _cluster_dbscan_batch(
    chr_name: str,
    X: np.ndarray,
    svtype: str,
    eps: int,
    min_samples: int,
    max_dup_inv_len: int,
) -> List:
    """Run one DBSCAN pass on a same-band candidate batch."""
    if X.size == 0:
        return []

    XX = np.array([[int(x[0]), int(x[1])] for x in X])
    clust = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    clust.fit(XX)

    cluster_dict = defaultdict(list)
    labels = clust.labels_

    for i, label in enumerate(labels):
        if label != -1:
            cluster_dict[label].append(X[i])
        else:
            cluster_dict[f"noise_{i}"].append(X[i])

    data_cluster = []
    for sig in cluster_dict.values():
        if len(sig) > 1:
            length_clusters = [sig] if svtype == 'INS' else cluster_by_length(sig)
        else:
            length_clusters = [sig]

        for length_cluster in length_clusters:
            if svtype == 'INS':
                length_cluster = _trim_ins_length_outliers(length_cluster, min_support=min_samples)
            data = np.array(length_cluster)

            if len(data) == 1:
                data_cluster.append([chr_name, data[0][0], data[0][2], len(data), svtype, './.', data[0][3], data[0][4]])
                continue

            data = sorted(data, key=lambda x: x[0])
            if svtype in {'DUP', 'INV'} and max_dup_inv_len > 0:
                data = [item for item in data if int(item[2]) >= 50 and int(item[2]) <= max_dup_inv_len]
            else:
                data = [item for item in data if int(item[2]) >= 50]

            if not data:
                continue

            first_elements = [int(item[0]) for item in data]
            third_elements = [int(item[2]) for item in data]
            start = math.ceil(median(first_elements))
            length = math.ceil(median(third_elements))
            readname_list = [item[3] for item in data]
            readtype = [item[4] for item in data]

            data_cluster.append([chr_name, start, length, len(data), svtype, './.', readname_list, readtype])

    return data_cluster


def mean_shift_def(
    chr_name: str,
    X: np.ndarray,
    svtype: str,
    error_mean: float,
    max_dup_inv_len: int = DEFAULT_MAX_DUP_INV_LEN,
    length_eps_bands: Optional[List[Tuple[int, int]]] = None,
) -> List:
    """Perform DBSCAN density-based clustering on SV candidates"""
    if X.size == 0:
        return []
    max_dup_inv_len = max(0, int(max_dup_inv_len))

    # Remove implausibly large DUP/INV candidates before clustering.
    if svtype in {'DUP', 'INV'} and max_dup_inv_len > 0:
        filtered = [row for row in X if int(row[2]) <= max_dup_inv_len]
        if not filtered:
            return []
        X = np.array(filtered, dtype=object)

    # Determine eps (neighborhood radius) based on SV type and error rate
    # Keep the same thresholds as before (bandwidth -> eps)
    if error_mean > 0.1:
        eps_map = {'DEL': 1000, 'INS': 300, 'DUP': 500, 'INV': 500}
        eps = eps_map.get(svtype, 1500)
    else:
        eps = 1500

    # Set min_samples for DBSCAN (minimum points to form a dense region)
    # Use 2 as default to allow pairs to form clusters
    min_samples = 2

    length_bands = _length_aware_eps_bands(svtype, error_mean, override_bands=length_eps_bands)
    if not length_bands:
        data_cluster = _cluster_dbscan_batch(
            chr_name=chr_name,
            X=X,
            svtype=svtype,
            eps=eps,
            min_samples=min_samples,
            max_dup_inv_len=max_dup_inv_len,
        )
        return sorted(data_cluster, key=lambda x: int(x[1]))

    rows = [list(row) for row in X]
    prev_max = -1
    data_cluster = []
    for max_len, band_eps in length_bands:
        band_rows = [row for row in rows if prev_max < int(row[2]) <= max_len]
        prev_max = max_len
        if not band_rows:
            continue
        band_array = np.array(band_rows, dtype=object)
        data_cluster.extend(
            _cluster_dbscan_batch(
                chr_name=chr_name,
                X=band_array,
                svtype=svtype,
                eps=band_eps,
                min_samples=min_samples,
                max_dup_inv_len=max_dup_inv_len,
            )
        )

    return sorted(data_cluster, key=lambda x: int(x[1]))


def cluster_translocations(candidates: List, max_distance: int = 1000) -> List:
    """Cluster translocation candidates"""
    sorted_candidates = sorted(candidates, key=lambda x: (x[1], x[0], x[3], x[4]))
    
    clusters = []
    data_cluster = []
    
    for candidate in sorted_candidates:
        start, read_name, current_direction, ref_chr_next, end, translocation_direction, variant_type, align_score, current_chr = candidate
        
        found_cluster = False
        for cluster in clusters:
            if (ref_chr_next == cluster[0][3] and current_chr == cluster[0][8] and
                abs(start - cluster[0][0]) <= max_distance and abs(end - cluster[0][4]) <= max_distance):
                cluster.append(candidate)
                found_cluster = True
                break
        
        if not found_cluster:
            clusters.append([candidate])
    
    for transs in clusters:
        if len(transs) == 1:
            data_cluster.append([
                transs[0][-1], transs[0][0], transs[0][2], transs[0][3], 
                transs[0][4], transs[0][5], 'BND', 1, '0/0', transs[0][1]
            ])
            continue
        
        first_chr_name = transs[0][-1]
        first_direction = transs[0][2]
        second_chr_name = transs[0][3]
        second_direction = transs[0][5]
        first_start_elements = [int(item[0]) for item in transs]
        start1 = math.ceil(median(first_start_elements))
        second_start_elements = [int(item[4]) for item in transs]
        start2 = math.ceil(median(second_start_elements))
        read_name_list = [item[1] for item in transs]
        
        data_cluster.append([
            first_chr_name, start1, first_direction, second_chr_name, 
            start2, second_direction, 'BND', len(transs), '0/0', read_name_list
        ])
    
    return sorted(data_cluster, key=lambda x: (x[0], x[1], x[3], x[4]))


# ============================================================================
# Genotype Functions
# ============================================================================

def likelihood_0_0(ref_depth: int, alt_depth: int, error_rate: float) -> float:
    """Calculate likelihood for 0/0 genotype"""
    p_ref = 1 - error_rate
    p_alt = error_rate
    return (p_ref ** ref_depth) * (p_alt ** alt_depth)


def likelihood_0_1(ref_depth: int, alt_depth: int, error_rate: float, bias_factor: float = 0.7) -> float:
    """Calculate likelihood for 0/1 genotype"""
    p_ref = bias_factor * (1 - error_rate)
    p_alt = (1 - bias_factor) * (1 - error_rate)
    return (p_ref ** ref_depth) * (p_alt ** alt_depth)


def likelihood_1_1(ref_depth: int, alt_depth: int, error_rate: float) -> float:
    """Calculate likelihood for 1/1 genotype"""
    p_ref = error_rate
    p_alt = 1 - error_rate
    return (p_ref ** ref_depth) * (p_alt ** alt_depth)


def bayesian_genotype_likelihood(ref_depth: int, alt_depth: int, error_rate: float,
                                  prior_AA: float, prior_AB: float, prior_BB: float) -> str:
    """Calculate genotype using Bayesian approach"""
    epsilon = 1e-10
    
    likelihood_AA = likelihood_0_0(ref_depth, alt_depth, error_rate) * prior_AA
    likelihood_AB = likelihood_0_1(ref_depth, alt_depth, error_rate) * prior_AB
    likelihood_BB = likelihood_1_1(ref_depth, alt_depth, error_rate) * prior_BB
    
    total_likelihood = max(likelihood_AA + likelihood_AB + likelihood_BB, epsilon)
    
    posterior_AA = likelihood_AA / total_likelihood
    posterior_AB = likelihood_AB / total_likelihood
    posterior_BB = likelihood_BB / total_likelihood
    
    return max([
        (posterior_AA, '0/0'), 
        (posterior_AB, '0/1'), 
        (posterior_BB, '1/1')
    ], key=lambda x: x[0])[1]


def em_genotype(ref_depth: int, alt_depth: int, error_rate: float, max_iter: int = 100) -> str:
    """Calculate genotype using EM algorithm"""
    p_00, p_01, p_11 = 0.3, 0.3, 0.3
    epsilon = 1e-50
    
    for _ in range(max_iter):
        likelihood_00 = (1 - error_rate) ** ref_depth * error_rate ** alt_depth * p_00
        likelihood_01 = (0.5 * (1 - error_rate)) ** ref_depth * (0.5 * (1 - error_rate)) ** alt_depth * p_01
        likelihood_11 = error_rate ** ref_depth * (1 - error_rate) ** alt_depth * p_11
        
        total_likelihood = max(likelihood_00 + likelihood_01 + likelihood_11, epsilon)
        
        p_00 = likelihood_00 / total_likelihood
        p_01 = likelihood_01 / total_likelihood
        p_11 = likelihood_11 / total_likelihood
    
    if p_00 > p_01 and p_00 > p_11:
        return '0/0'
    elif p_01 > p_00 and p_01 > p_11:
        return '0/1'
    return '1/1'


def genotype_by_depth_ratio(ref_depth: int, alt_depth: int, threshold: float = 0.7) -> str:
    """Calculate genotype based on depth ratio"""
    total_depth = ref_depth + alt_depth
    if total_depth == 0:
        return './.'
    
    ref_ratio = ref_depth / total_depth
    
    if ref_ratio >= threshold:
        return '0/0'
    elif ref_ratio <= (1 - threshold):
        return '1/1'
    return '0/1'


def combined_genotype_voting(ref_depth: int, alt_depth: int, error_rate: float,
                             prior_AA: float, prior_AB: float, prior_BB: float) -> str:
    """Combine multiple genotyping methods using voting"""
    bayesian_geno = bayesian_genotype_likelihood(ref_depth, alt_depth, error_rate, prior_AA, prior_AB, prior_BB)
    em_geno = em_genotype(ref_depth, alt_depth, error_rate)
    depth_ratio_geno = genotype_by_depth_ratio(ref_depth, alt_depth, 0.8)
    
    votes = Counter([bayesian_geno, em_geno, depth_ratio_geno])
    return votes.most_common(1)[0][0]


def threshold_ref_count(num: int) -> int:
    """Calculate reference count threshold"""
    if num <= 2:
        return 20 * num
    elif 3 <= num <= 5:
        return 9 * num
    elif 6 <= num <= 15:
        return 7 * num
    return 5 * num


def _safe_log(value: float, eps: float = 1e-300) -> float:
    """Safe log to avoid -inf from numerical underflow."""
    return math.log(max(float(value), eps))


def _clamp_error_rate(error_rate: float, eps: float = 1e-6) -> float:
    """Clamp error rate into a numerically stable range."""
    return min(max(float(error_rate), eps), 1.0 - eps)


def log_likelihood_0_0(ref_depth: int, alt_depth: int, error_rate: float) -> float:
    """Log-likelihood for 0/0 genotype."""
    e = _clamp_error_rate(error_rate)
    return ref_depth * _safe_log(1.0 - e) + alt_depth * _safe_log(e)


def log_likelihood_0_1(ref_depth: int, alt_depth: int, error_rate: float, bias_factor: float = 0.5) -> float:
    """Log-likelihood for 0/1 genotype with optional allelic bias."""
    e = _clamp_error_rate(error_rate)
    b = min(max(float(bias_factor), 0.05), 0.95)
    # Include sequencing error into heterozygous emission probabilities.
    p_ref = b * (1.0 - e) + (1.0 - b) * e
    p_alt = (1.0 - b) * (1.0 - e) + b * e
    return ref_depth * _safe_log(p_ref) + alt_depth * _safe_log(p_alt)


def log_likelihood_1_1(ref_depth: int, alt_depth: int, error_rate: float) -> float:
    """Log-likelihood for 1/1 genotype."""
    e = _clamp_error_rate(error_rate)
    return ref_depth * _safe_log(e) + alt_depth * _safe_log(1.0 - e)


def genotype_by_log_gl(
    ref_depth: int,
    alt_depth: int,
    error_rate: float,
    prior_AA: float = 1 / 3,
    prior_AB: float = 1 / 3,
    prior_BB: float = 1 / 3,
    bias_factor: float = 0.5,
) -> str:
    """
    Genotype by log-space genotype likelihoods.
    This is numerically stable for high coverage depth.
    """
    if ref_depth + alt_depth <= 0:
        return './.'

    log_scores = {
        '0/0': _safe_log(prior_AA) + log_likelihood_0_0(ref_depth, alt_depth, error_rate),
        '0/1': _safe_log(prior_AB) + log_likelihood_0_1(ref_depth, alt_depth, error_rate, bias_factor=bias_factor),
        '1/1': _safe_log(prior_BB) + log_likelihood_1_1(ref_depth, alt_depth, error_rate),
    }
    return max(log_scores.items(), key=lambda x: x[1])[0]


# ============================================================================
# Main Detection Functions
# ============================================================================

def merge_intervals(intervals: List) -> List:
    """Merge overlapping intervals"""
    if not intervals:
        return []
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[1] == last[2]:
            last[2] = current[2]
        else:
            merged.append(current)
    return merged


def mergedeleton_long(
    pre: np.ndarray,
    bamfile,
    ssstart: int,
    chr_name: str,
    index: np.ndarray,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    max_dup_inv_len: int = DEFAULT_MAX_DUP_INV_LEN,
) -> Tuple[List, List]:
    """Process prediction results and extract SV candidates"""
    data = []
    candidate = []
    breakpointall = []
    
    for i, pred in enumerate(pre):
        if pred >= decision_threshold:
            data.append([chr_name, index[i], index[i] + 2000])
        ssstart += 2000
    
    if not data:
        return [], []
    
    data = merge_intervals(data)
    
    for chr_name, start, end in data:
        for read in bamfile.fetch(chr_name, start, end):
            nm = read.get_tag('NM') if read.has_tag('NM') else None
            
            CIGAR_DEL = 2
            CIGAR_INS = 1
            CIGAR_CLIP = [4, 5]
            K_MM = 1.0
            
            indel = [b for a, b in read.cigartuples if a in [CIGAR_INS, CIGAR_DEL]]
            indel_bases = sum(indel)
            total_segment_length = sum([b for a, b in read.cigartuples if a not in CIGAR_CLIP + [CIGAR_DEL]])
            if total_segment_length <= 0:
                continue

            # Some BAMs omit NM. Fall back to an indel-only approximation so
            # inference can continue instead of aborting on missing tags.
            nm_value = indel_bases if nm is None else int(nm)
            num_of_mismatch = max(nm_value - indel_bases, 0)
            
            mm_rate = num_of_mismatch * K_MM / total_segment_length if total_segment_length > 0 else 0
            error_rate = nm_value * K_MM / total_segment_length if total_segment_length > 0 else 0
            
            alignscore = alignment_quality_score(read.mapq, error_rate, num_of_mismatch, mm_rate)
            
            if read.is_supplementary or read.is_secondary or read.is_unmapped:
                continue
            
            read_candidates = []
            if read.mapq >= 20:
                cigarread(read, read_candidates, start, end, alignscore)
                if read.has_tag('SA'):
                    split_read = splitreadlist(read)
                    analyze_read_segments(
                        read,
                        feature_read_segment(split_read),
                        read_candidates,
                        start,
                        end,
                        alignscore,
                        max_dup_inv_len=max_dup_inv_len,
                    )
            candidate.extend(merge_read_level_ins_candidates(read_candidates))
            
            breakpointall.append([error_rate, read.reference_name])
    
    return candidate, breakpointall


def load_all_data(
    predict_path: str,
    data_store_path: str,
    start: int,
    end: int,
    chr_name: str,
    bamfilepath: str,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    max_dup_inv_len: int = DEFAULT_MAX_DUP_INV_LEN,
) -> Tuple[List, List]:
    """Load prediction data and extract candidates"""
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=1)
    try:
        try:
            x_t = np.load(predict_path)
            xindex = load_feature_index(data_store_path)
        except (FileNotFoundError, OSError, ValueError):
            return [], []
        
        predict1 = x_t.flatten()
        base = np.array(predict1)
        
        if len(np.where(base >= decision_threshold)[0]) == 0:
            return [], []
        
        candidate, breakpointall = mergedeleton_long(
            base,
            bamfile,
            start,
            chr_name,
            xindex,
            decision_threshold=decision_threshold,
            max_dup_inv_len=max_dup_inv_len,
        )
        return candidate, breakpointall
    finally:
        bamfile.close()


def analysis_candidate(
    candidatem: List,
    chr_name: str,
    max_dup_inv_len: int = DEFAULT_MAX_DUP_INV_LEN,
    ins_eps_bands: Optional[List[Tuple[int, int]]] = None,
    inv_eps_bands: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List, List]:
    """Analyze candidates and cluster by SV type"""
    resultlist = []
    error_mean_all = []
    signal_dict = defaultdict(list)
    signal_dict_error = defaultdict(list)
    
    for sv_candidate in candidatem:
        for sv_candi in sv_candidate[0]:
            if sv_candi[-1] == chr_name:
                signal_dict[chr_name].append(sv_candi)
        for sv_candi in sv_candidate[1]:
            if sv_candi[-1] == chr_name:
                signal_dict_error[chr_name].append(sv_candi)
    
    for chrname, sv_candidate in signal_dict.items():
        del_signal, ins_signal, inv_signal, dup_signal, tra_signal = [], [], [], [], []
        
        error_data = signal_dict_error[chrname]
        error_mean = np.mean([float(item[0]) for item in error_data]) if error_data else 0
        
        sv_candidate = sorted(sv_candidate, key=lambda x: x[0])
        
        for sv_candi in sv_candidate:
            sv_type = sv_candi[-3]
            if sv_type == 'DEL':
                del_signal.append([sv_candi[0], sv_candi[1] + sv_candi[0], sv_candi[1], 
                                   sv_candi[2], sv_candi[4], sv_candi[6]])
            elif sv_type == 'INS':
                ins_signal.append([sv_candi[0], sv_candi[0] + 1, sv_candi[1], 
                                   sv_candi[2], sv_candi[4], sv_candi[6]])
            elif sv_type == 'INV':
                inv_signal.append([sv_candi[0], sv_candi[1] + sv_candi[0], sv_candi[1], 
                                   sv_candi[2], sv_candi[4], sv_candi[6]])
            elif sv_type == 'DUP':
                dup_signal.append([sv_candi[0], sv_candi[1] + sv_candi[0], sv_candi[1], 
                                   sv_candi[2], sv_candi[4], sv_candi[6]])
            elif sv_type == 'TRA':
                tra_signal.append(sv_candi)
        
        ins_signal = np.array(sorted(ins_signal, key=lambda x: x[0])) if ins_signal else np.array([])
        del_signal = np.array(sorted(del_signal, key=lambda x: x[0])) if del_signal else np.array([])
        inv_signal = np.array(sorted(inv_signal, key=lambda x: x[0])) if inv_signal else np.array([])
        dup_signal = np.array(sorted(dup_signal, key=lambda x: x[0])) if dup_signal else np.array([])
        tra_signal = sorted(tra_signal, key=lambda x: (x[8], x[0], x[3], x[4]))
        
        sum_del = mean_shift_def(chrname, del_signal, 'DEL', error_mean, max_dup_inv_len=max_dup_inv_len)
        sum_ins = mean_shift_def(chrname, ins_signal, 'INS', error_mean, max_dup_inv_len=max_dup_inv_len, length_eps_bands=ins_eps_bands)
        sum_inv = mean_shift_def(chrname, inv_signal, 'INV', error_mean, max_dup_inv_len=max_dup_inv_len, length_eps_bands=inv_eps_bands)
        sum_dup = mean_shift_def(chrname, dup_signal, 'DUP', error_mean, max_dup_inv_len=max_dup_inv_len)
        sum_trans = cluster_translocations(tra_signal)
        
        resultlist.extend(sum_del + sum_ins + sum_inv + sum_dup + sum_trans)
        error_mean_all.append(error_mean)
    
    return resultlist, error_mean_all


def average_read_coverage(bamfilepath: str, chr_name: str, lengthh: int) -> int:
    """Calculate average read coverage for a chromosome"""
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=1)
    chr_align_length = 0
    
    for read in bamfile.fetch(chr_name):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        chr_align_length += read.query_length
    
    bamfile.close()
    return chr_align_length


def _strip_chr_prefix(contig: str) -> str:
    contig_name = str(contig)
    return contig_name[3:] if contig_name.startswith('chr') else contig_name


def prediction_output_path(testpath: str, contig: str, start: int, end: int) -> str:
    """Build the on-disk prediction filename for one 10 Mb region."""
    contig_name = _strip_chr_prefix(contig)
    return os.path.join(testpath, f"chr{contig_name}_{int(start)}_{int(end)}_predict.npy")


def _resolve_contigs_and_lengths(
    bamfilepath: str,
    contigg: List,
    bam_threads: int = 1,
) -> Tuple[List[str], Dict[str, int]]:
    """Resolve requested contigs against the BAM index and collect lengths."""
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=max(1, int(bam_threads)))
    try:
        stats = list(bamfile.get_index_statistics())
        contig2length = {}
        for idx, stat in enumerate(stats):
            if idx < len(bamfile.lengths):
                contig2length[str(stat.contig)] = int(bamfile.lengths[idx])

        if not contigg:
            resolved_contigs = [str(stat.contig) for stat in stats]
        else:
            resolved_contigs = []
            for contig in np.asarray(contigg).astype(str):
                contig_name = str(contig)
                if contig_name not in contig2length:
                    logger.warning("Skipping unknown contig %s (not found in BAM index)", contig_name)
                    continue
                resolved_contigs.append(contig_name)
        return resolved_contigs, contig2length
    finally:
        bamfile.close()


def collect_prediction_jobs(
    bamfilepath: str,
    data_path: str,
    testpath: str,
    contigg: List,
) -> Tuple[List[str], Dict[str, int], List[Dict[str, Any]]]:
    """Discover feature-store backed 10 Mb regions and their prediction targets."""
    resolved_contigs, contig2length = _resolve_contigs_and_lengths(bamfilepath, contigg, bam_threads=1)
    jobs: List[Dict[str, Any]] = []

    for chr_name in resolved_contigs:
        chr_length = contig2length.get(chr_name)
        if chr_length is None:
            continue
        data_contig = _strip_chr_prefix(chr_name)
        for start in range(0, int(chr_length), REGION_SIZE):
            end = start + REGION_SIZE
            data_store_path = resolve_feature_store_path(data_path, data_contig, start, end)
            if data_store_path is None:
                continue
            jobs.append({
                'chr_name': chr_name,
                'start': int(start),
                'end': int(end),
                'data_store_path': str(data_store_path),
                'predict_path': prediction_output_path(testpath, chr_name, start, end),
            })

    return resolved_contigs, contig2length, jobs


def summarize_prediction_jobs(jobs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Split jobs into reusable predictions, missing predictions, and empty feature stores."""
    ready_jobs: List[Dict[str, Any]] = []
    missing_jobs: List[Dict[str, Any]] = []
    empty_store_count = 0

    for job in jobs:
        try:
            index = load_feature_index(job['data_store_path'])
        except (FileNotFoundError, OSError, ValueError):
            missing_jobs.append(job)
            continue

        if int(np.asarray(index).size) == 0:
            empty_store_count += 1
            continue

        try:
            np.load(job['predict_path'], mmap_mode='r')
            ready_jobs.append(job)
        except (FileNotFoundError, OSError, ValueError):
            missing_jobs.append(job)

    return ready_jobs, missing_jobs, empty_store_count


# ============================================================================
# PyTorch Model Prediction
# ============================================================================

def model_predict(
    weights_path: str,
    bamfilepath: str,
    data_path: str,
    testpath: str,
    contigg: List,
    device: str = 'cuda',
    model_type: str = 'mamba',
    platform: Optional[str] = None,
    predict_batch_size: int = 64,
    predict_head: str = 'cnn_mamba',
    reuse_existing: bool = False,
):
    """
    Predict SV candidates using PyTorch model

    Args:
        weights_path: Path to model weights
        bamfilepath: Path to BAM file
        data_path: Path to feature data directory
        testpath: Path to save predictions
        contigg: List of contigs to process
        device: Device for inference ('cuda' or 'cpu')
        model_type: The model architecture to use (only 'mamba' is supported)
        platform: Sequencing platform name (ccs/clr/ont); auto-inferred when omitted
        predict_head: Prediction head to use: cnn, mamba, or cnn_mamba
    """
    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    device = torch.device(device)
    feature_config = resolve_feature_config(data_dir=data_path, weights_path=weights_path)
    platform_name = resolve_platform_name(platform, bamfilepath, data_path)
    platform_id = resolve_platform_id(platform, bamfilepath, data_path)
    
    # Load model
    logger.info(f"Loading model ({model_type}) from {weights_path}")
    logger.info(
        "Feature config for inference: version=%s dim=%d window=%d (source=%s)",
        feature_config['feature_version'],
        feature_config['feature_dim'],
        feature_config['window_size'],
        feature_config['source'],
    )
    logger.info("Inference platform: %s (id=%d)", platform_name, platform_id)
    logger.info("Inference prediction head: %s", predict_head)
    model = create_model(
        model_type=model_type,
        pretrained_weights=weights_path,
        device=device,
        feature_dim=int(feature_config['feature_dim']),
        window_size=int(feature_config['window_size']),
    )

    # Enable multi-GPU inference with DataParallel
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for inference with DataParallel")
        model = nn.DataParallel(model)
    else:
        logger.info("Using single GPU or CPU for inference")

    model.eval()
    
    contigg, _, prediction_jobs = collect_prediction_jobs(
        bamfilepath=bamfilepath,
        data_path=data_path,
        testpath=testpath,
        contigg=contigg,
    )
    if not prediction_jobs:
        logger.warning("No feature-store regions found under %s for prediction", data_path)
        return

    logger.info(f"Processing contigs: {contigg}")

    jobs_by_contig: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job in prediction_jobs:
        jobs_by_contig[str(job['chr_name'])].append(job)

    batch_size = max(1, int(predict_batch_size))
    for chr_name in contigg:
        contig_jobs = jobs_by_contig.get(str(chr_name), [])
        logger.info(f"Processing chromosome {chr_name} ({len(contig_jobs)} segments)")

        for seg_idx, job in enumerate(contig_jobs, start=1):
            logger.info(f"CMSV predict {chr_name}: {seg_idx}/{len(contig_jobs)}")

            store_path = job['data_store_path']
            save_path = job['predict_path']
            if reuse_existing:
                try:
                    np.load(save_path, mmap_mode='r')
                    logger.info("Skipping existing prediction %s", save_path)
                    continue
                except (FileNotFoundError, OSError, ValueError):
                    pass

            try:
                x_t = load_feature_data(store_path, mmap_mode='r')
            except (ValueError, FileNotFoundError, OSError):
                continue

            if x_t.size == 0:
                continue

            if x_t.ndim == 2:
                total_samples = int(x_t.shape[0]) // int(feature_config['window_size'])
            else:
                total_samples = int(x_t.shape[0])

            if total_samples == 0:
                continue

            predict1 = np.empty(total_samples, dtype=np.float32)
            with torch.inference_mode():
                for batch_start in range(0, total_samples, batch_size):
                    batch_end = min(batch_start + batch_size, total_samples)
                    if x_t.ndim == 2:
                        row_start = batch_start * int(feature_config['window_size'])
                        row_end = batch_end * int(feature_config['window_size'])
                        batch_np = np.asarray(x_t[row_start:row_end], dtype=np.float32)
                    else:
                        batch_np = np.asarray(x_t[batch_start:batch_end], dtype=np.float32)
                    if batch_np.ndim == 2:
                        batch_np = batch_np.reshape(
                            -1,
                            int(feature_config['window_size']),
                            int(feature_config['feature_dim']),
                        )
                    elif batch_np.ndim != 3:
                        raise ValueError(f"Unsupported feature array shape for {store_path}: {batch_np.shape}")

                    batch_data = torch.from_numpy(batch_np).to(device)
                    batch_platform_ids = torch.full(
                        (batch_data.shape[0],),
                        int(platform_id),
                        dtype=torch.long,
                        device=device,
                    )
                    output = model(batch_data, platform_ids=batch_platform_ids, return_aux=True)
                    if isinstance(output, dict):
                        selected_output = output.get(f'{predict_head}_pred')
                        if selected_output is None and predict_head == 'cnn_mamba':
                            selected_output = output.get('mamba_pred')
                        if selected_output is None:
                            raise KeyError(
                                f"Requested predict_head={predict_head!r} is unavailable in model outputs: "
                                f"{sorted(output.keys())}"
                            )
                    else:
                        selected_output = output
                    batch_pred = np.asarray(selected_output.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                    if batch_pred.shape[0] != (batch_end - batch_start):
                        raise ValueError(
                            f"Unexpected prediction shape for {store_path}: got {batch_pred.shape}, "
                            f"expected {(batch_end - batch_start,)}"
                        )
                    predict1[batch_start:batch_end] = batch_pred

            del x_t
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            np.save(save_path, predict1)
    
    logger.info("Prediction completed")


# ============================================================================
# VCF Generation Functions
# ============================================================================

def _write_vcf_header(handle, contiglength: Dict):
    handle.write("##fileformat=VCFv4.2\n")
    handle.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
    for contig in contiglength:
        handle.write(f"##contig=<ID={contig},length={int(contiglength[contig])}>\n")
    handle.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">\n')
    handle.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, INS=Insertion, INV=Inversion, DUP=Duplication, BND=Translocation">\n')
    handle.write('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">\n')
    handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t.\n")


def generate_vcf(inslist: List, contiglength: Dict, outvcfpath: str, BND_result: Optional[List] = None):
    """Generate VCF file from SV results"""
    tmp_path = f"{outvcfpath}.tmp"
    skipped_main = 0
    skipped_bnd = 0

    try:
        with open(tmp_path, "w") as f:
            _write_vcf_header(f, contiglength)

            for rec in inslist:
                try:
                    contig = str(rec[0])
                    pos = int(rec[1])
                    svlen = int(rec[2])
                    svtype = str(rec[4])
                    geno = str(rec[5]) if len(rec) > 5 else "./."
                    if svtype == 'INS':
                        end = pos
                    elif svtype in ['DEL', 'INV', 'DUP']:
                        end = pos + svlen
                    else:
                        continue
                    if pos < 0 or end < 0:
                        raise ValueError(f"negative coordinate pos={pos} end={end}")
                    info = f"SVLEN={svlen};SVTYPE={svtype};END={end};"
                    f.write(f"{contig}\t{pos}\t.\tN\t<{svtype}>\t.\tPASS\t{info}\tGT\t{geno}\n")
                except Exception as exc:
                    skipped_main += 1
                    if skipped_main <= 5:
                        logger.warning("Skipping malformed VCF record %s: %s", rec, exc)

            if BND_result:
                for transs in BND_result:
                    for trans in transs:
                        try:
                            first_chr, start1, first_direction, second_chr, start2, second_direction, svtype, support_read, genotype, _, _ = trans
                            start1 = int(start1)
                            start2 = int(start2)
                            geno = str(genotype) if genotype else "./."

                            if first_direction == 'fwd' and second_direction == 'fwd':
                                alt1 = f'N[{second_chr}:{start2}['
                                alt2 = f']{first_chr}:{start1}]N'
                            elif first_direction == 'fwd' and second_direction == 'rev':
                                alt1 = f'N]{second_chr}:{start2}]'
                                alt2 = f'[{first_chr}:{start1}[N'
                            elif first_direction == 'rev' and second_direction == 'fwd':
                                alt1 = f']{second_chr}:{start2}]N'
                                alt2 = f'N[{first_chr}:{start1}['
                            else:
                                alt1 = f'[{second_chr}:{start2}[N'
                                alt2 = f'N]{first_chr}:{start1}]'

                            info = 'SVTYPE=BND;'
                            f.write(f'{first_chr}\t{start1}\t.\tN\t{alt1}\t.\tPASS\t{info}\tGT\t{geno}\n')
                            f.write(f'{second_chr}\t{start2}\t.\tN\t{alt2}\t.\tPASS\t{info}\tGT\t{geno}\n')
                        except Exception as exc:
                            skipped_bnd += 1
                            if skipped_bnd <= 5:
                                logger.warning("Skipping malformed BND record %s: %s", trans, exc)

        os.replace(tmp_path, outvcfpath)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if skipped_main or skipped_bnd:
        logger.warning(
            "VCF generation skipped %d malformed SV records and %d malformed BND records for %s",
            skipped_main,
            skipped_bnd,
            outvcfpath,
        )


def generate_vcf_bnd(inslist: List, contiglength: Dict, outvcfpath: str, BND_result: Optional[List] = None):
    """Generate VCF file for BND/Translocation results only"""
    tmp_path = f"{outvcfpath}.tmp"
    skipped_bnd = 0
    try:
        with open(tmp_path, "w") as f:
            _write_vcf_header(f, contiglength)
            if BND_result:
                for transs in BND_result:
                    for trans in transs:
                        try:
                            first_chr, start1, first_direction, second_chr, start2, second_direction, svtype, support_read, genotype, _, _ = trans
                            start1 = int(start1)
                            start2 = int(start2)
                            geno = './.'

                            if first_direction == 'fwd' and second_direction == 'fwd':
                                alt1 = f'N[{second_chr}:{start2}['
                                alt2 = f']{first_chr}:{start1}]N'
                            elif first_direction == 'fwd' and second_direction == 'rev':
                                alt1 = f'N]{second_chr}:{start2}]'
                                alt2 = f'[{first_chr}:{start1}[N'
                            elif first_direction == 'rev' and second_direction == 'fwd':
                                alt1 = f']{second_chr}:{start2}]N'
                                alt2 = f'N[{first_chr}:{start1}['
                            else:
                                alt1 = f'[{second_chr}:{start2}[N'
                                alt2 = f'N]{first_chr}:{start1}]'

                            info = 'SVTYPE=BND;'
                            f.write(f'{first_chr}\t{start1}\t.\tN\t{alt1}\t.\tPASS\t{info}\tGT\t{geno}\n')
                            f.write(f'{second_chr}\t{start2}\t.\tN\t{alt2}\t.\tPASS\t{info}\tGT\t{geno}\n')
                        except Exception as exc:
                            skipped_bnd += 1
                            if skipped_bnd <= 5:
                                logger.warning("Skipping malformed BND-only record %s: %s", trans, exc)
        os.replace(tmp_path, outvcfpath)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if skipped_bnd:
        logger.warning(
            "BND-only VCF generation skipped %d malformed BND records for %s",
            skipped_bnd,
            outvcfpath,
        )


def _dump_vcf_debug_snapshot(
    debug_path: str,
    all_result: List,
    BND_result: List,
    contig2length: Dict,
):
    """Persist VCF-generation inputs for postmortem debugging."""
    with open(debug_path, "wb") as handle:
        pickle.dump(
            {
                "all_result": all_result,
                "BND_result": BND_result,
                "contig2length": contig2length,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# ============================================================================
# Breakpoint Refinement Functions (Post-processing)
# ============================================================================

def _parse_refine_types(refine_types: str) -> Set[str]:
    """Parse refinement SV types from comma-separated string."""
    if not refine_types:
        return set()
    return {item.strip().upper() for item in refine_types.split(',') if item.strip()}


def _extract_indel_events_from_read(read, min_sv_size: int = 40):
    """
    Extract INS/DEL events from one read.
    Returns:
      ins_events: List[(pos, length, read_name)]
      del_events: List[(start, end, length, read_name)]
    """
    ins_events = []
    del_events = []
    if read.cigartuples is None:
        return ins_events, del_events

    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op in [0, 7, 8]:  # M, =, X
            ref_pos += length
        elif op == 1:  # INS
            if length >= min_sv_size:
                ins_events.append((ref_pos, int(length), read.query_name))
        elif op == 2:  # DEL
            start = ref_pos
            end = ref_pos + length
            if length >= min_sv_size:
                del_events.append((int(start), int(end), int(length), read.query_name))
            ref_pos = end
        elif op == 3:  # N
            ref_pos += length
        else:
            # S/H/P and other operations do not consume reference in this context.
            continue

    return ins_events, del_events


def _collect_local_indel_events(
    bamfile,
    contig: str,
    region_start: int,
    region_end: int,
    min_mapq: int = 20,
    min_sv_size: int = 40,
    max_reads: int = 5000,
):
    """Collect local INS/DEL events from BAM reads in a region."""
    ins_events = []
    del_events = []

    try:
        contig_len = bamfile.get_reference_length(contig)
    except (ValueError, KeyError):
        return ins_events, del_events

    start = max(0, int(region_start))
    end = min(contig_len, int(region_end))
    if end <= start:
        end = min(contig_len, start + 1)

    seen = 0
    try:
        read_iter = bamfile.fetch(contig, start, end)
    except ValueError:
        return ins_events, del_events

    for read in read_iter:
        if seen >= max_reads:
            break
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue

        seen += 1
        ins, dele = _extract_indel_events_from_read(read, min_sv_size=min_sv_size)
        ins_events.extend(ins)
        del_events.extend(dele)

    return ins_events, del_events


def _refine_ins_record(
    record: List,
    bamfile,
    window_bp: int,
    min_evidence: int,
    max_shift: int,
    min_len_ratio: float,
    max_len_ratio: float,
):
    """Refine INS breakpoint using local CIGAR insertion evidence."""
    contig = str(record[0])
    pred_pos = int(record[1])
    pred_len = max(1, int(record[2]))

    ins_events, _ = _collect_local_indel_events(
        bamfile=bamfile,
        contig=contig,
        region_start=pred_pos - window_bp,
        region_end=pred_pos + window_bp + 1,
    )

    candidate_events = []
    for pos, ins_len, read_name in ins_events:
        if abs(int(pos) - pred_pos) > (window_bp + max_shift):
            continue
        ratio = float(ins_len) / float(pred_len)
        # Loose pre-filter before strict acceptance check.
        if ratio < 0.25 or ratio > 4.0:
            continue
        candidate_events.append((int(pos), int(ins_len), read_name))

    if not candidate_events:
        return record, {'changed': False, 'reason': 'no_event', 'evidence': 0}

    evidence_count = len({x[2] for x in candidate_events})
    if evidence_count < min_evidence:
        return record, {'changed': False, 'reason': 'low_evidence', 'evidence': evidence_count}

    refined_pos = int(round(float(np.median([x[0] for x in candidate_events]))))
    refined_len = int(round(float(np.median([x[1] for x in candidate_events]))))
    if refined_len < 40:
        return record, {'changed': False, 'reason': 'short_len', 'evidence': evidence_count}

    shift = abs(refined_pos - pred_pos)
    len_ratio = float(refined_len) / float(pred_len)

    if shift > max_shift:
        return record, {'changed': False, 'reason': 'over_shift', 'evidence': evidence_count}
    if len_ratio < min_len_ratio or len_ratio > max_len_ratio:
        return record, {'changed': False, 'reason': 'len_ratio', 'evidence': evidence_count}

    new_record = list(record)
    new_record[1] = refined_pos
    new_record[2] = refined_len
    changed = (new_record[1] != pred_pos) or (new_record[2] != pred_len)

    return new_record, {
        'changed': changed,
        'reason': 'updated' if changed else 'same_value',
        'evidence': evidence_count,
        'shift': shift,
        'len_ratio': len_ratio,
    }


def _refine_del_record(
    record: List,
    bamfile,
    window_bp: int,
    min_evidence: int,
    max_shift: int,
    min_len_ratio: float,
    max_len_ratio: float,
):
    """Refine DEL breakpoint using local CIGAR deletion evidence."""
    contig = str(record[0])
    pred_start = int(record[1])
    pred_len = max(1, int(record[2]))
    pred_end = pred_start + pred_len

    _, del_events = _collect_local_indel_events(
        bamfile=bamfile,
        contig=contig,
        region_start=pred_start - window_bp,
        region_end=pred_end + window_bp,
    )

    candidate_events = []
    for start, end, del_len, read_name in del_events:
        overlap = max(0, min(end, pred_end) - max(start, pred_start))
        overlap_ratio = float(overlap) / float(max(1, min(del_len, pred_len)))
        near_start = abs(int(start) - pred_start) <= (window_bp + max_shift)
        near_end = abs(int(end) - pred_end) <= (window_bp + max_shift)
        if not (near_start or near_end or overlap_ratio >= 0.2):
            continue
        ratio = float(del_len) / float(pred_len)
        if ratio < 0.25 or ratio > 4.0:
            continue
        candidate_events.append((int(start), int(end), int(del_len), read_name))

    if not candidate_events:
        return record, {'changed': False, 'reason': 'no_event', 'evidence': 0}

    evidence_count = len({x[3] for x in candidate_events})
    if evidence_count < min_evidence:
        return record, {'changed': False, 'reason': 'low_evidence', 'evidence': evidence_count}

    refined_start = int(round(float(np.median([x[0] for x in candidate_events]))))
    refined_end = int(round(float(np.median([x[1] for x in candidate_events]))))
    refined_len = max(1, refined_end - refined_start)

    if refined_len < 40:
        return record, {'changed': False, 'reason': 'short_len', 'evidence': evidence_count}

    shift = max(abs(refined_start - pred_start), abs(refined_end - pred_end))
    len_ratio = float(refined_len) / float(pred_len)

    if shift > max_shift:
        return record, {'changed': False, 'reason': 'over_shift', 'evidence': evidence_count}
    if len_ratio < min_len_ratio or len_ratio > max_len_ratio:
        return record, {'changed': False, 'reason': 'len_ratio', 'evidence': evidence_count}

    new_record = list(record)
    new_record[1] = refined_start
    new_record[2] = refined_len
    changed = (new_record[1] != pred_start) or (new_record[2] != pred_len)

    return new_record, {
        'changed': changed,
        'reason': 'updated' if changed else 'same_value',
        'evidence': evidence_count,
        'shift': shift,
        'len_ratio': len_ratio,
    }


def breakpoint_refine(
    records: List,
    bamfile,
    enabled: bool = False,
    refine_types: str = 'INS,DEL',
    window_bp: int = 200,
    min_evidence: int = 2,
    max_shift: int = 500,
    min_len_ratio: float = 0.5,
    max_len_ratio: float = 2.0,
    ins_min_evidence: Optional[int] = None,
    ins_max_shift: Optional[int] = None,
    ins_min_len_ratio: Optional[float] = None,
    ins_max_len_ratio: Optional[float] = None,
):
    """
    Refine SV breakpoints as a post-processing stage.
    Only INS/DEL are refined in this version; unsupported types are untouched.
    """
    if not enabled:
        return records, []

    target_types = _parse_refine_types(refine_types)
    if not target_types:
        return records, []

    refined_records = []
    summary_rows = []

    for idx, record in enumerate(records):
        svtype = str(record[4]).upper()
        new_record = record
        meta = {'changed': False, 'reason': 'type_skip', 'evidence': 0}

        if svtype in target_types:
            if svtype == 'INS':
                cur_min_evidence = int(ins_min_evidence) if ins_min_evidence is not None else int(min_evidence)
                cur_max_shift = int(ins_max_shift) if ins_max_shift is not None else int(max_shift)
                cur_min_len_ratio = float(ins_min_len_ratio) if ins_min_len_ratio is not None else float(min_len_ratio)
                cur_max_len_ratio = float(ins_max_len_ratio) if ins_max_len_ratio is not None else float(max_len_ratio)
                new_record, meta = _refine_ins_record(
                    record=record,
                    bamfile=bamfile,
                    window_bp=window_bp,
                    min_evidence=cur_min_evidence,
                    max_shift=cur_max_shift,
                    min_len_ratio=cur_min_len_ratio,
                    max_len_ratio=cur_max_len_ratio,
                )
            elif svtype == 'DEL':
                new_record, meta = _refine_del_record(
                    record=record,
                    bamfile=bamfile,
                    window_bp=window_bp,
                    min_evidence=min_evidence,
                    max_shift=max_shift,
                    min_len_ratio=min_len_ratio,
                    max_len_ratio=max_len_ratio,
                )
            else:
                meta = {'changed': False, 'reason': 'not_supported', 'evidence': 0}

        refined_records.append(new_record)
        summary_rows.append([
            idx,
            str(record[0]),
            int(record[1]),
            int(record[2]),
            int(new_record[1]),
            int(new_record[2]),
            svtype,
            int(meta.get('evidence', 0)),
            int(meta.get('changed', False)),
            str(meta.get('reason', 'na')),
        ])

    return refined_records, summary_rows


def write_refine_summary(summary_rows: List, output_path: str):
    """Write breakpoint refinement summary as TSV."""
    if not summary_rows:
        return
    header = "index\tchrom\torig_pos\torig_len\tnew_pos\tnew_len\tsvtype\tevidence\tchanged\treason\n"
    with open(output_path, 'w') as f:
        f.write(header)
        for row in summary_rows:
            f.write("\t".join(map(str, row)) + "\n")


# ============================================================================
# Support Read Calculation (Multi-threaded)
# ============================================================================

def _adjust_min_support(min_support: int, support_scale: float = 1.0, support_cap: int = 0) -> int:
    """Apply lightweight inference-time calibration to support thresholds."""
    scaled = max(1, int(round(float(min_support) * float(support_scale))))
    if int(support_cap) > 0:
        scaled = min(scaled, int(support_cap))
    return max(1, scaled)


def support_read_calculate_multi(
    data: List,
    bamfile_path: str,
    globle_coverage: int,
    error_rate1: float,
    support_scale: float = 1.0,
    support_cap: int = 0,
    ins_support_scale: float = 1.0,
    ins_min_support_floor: int = 0,
    ins_short_len: int = 300,
    ins_short_support_bonus: int = 0,
    ins_long_len: int = 0,
    ins_long_support_delta: int = 0,
    inv_support_scale: float = 1.0,
    inv_min_support_floor: int = 0,
    inv_short_len: int = 3000,
    inv_short_support_bonus: int = 0,
) -> List:
    """Calculate minimum support for SV candidates (multi-threaded version)"""
    try:
        error_rate1 = error_rate1[0] if isinstance(error_rate1, (list, tuple)) else error_rate1
    except:
        error_rate1 = 0
    
    bamfile = pysam.AlignmentFile(bamfile_path, 'rb', threads=1)
    
    for message in data:
        svtype = message[4]
        contig = message[0]
        
        if svtype == 'INS':
            start_region = max(0, int(message[1]) - 1000)
            end_region = int(message[1]) + 1000
        elif svtype in ['DEL', 'INV', 'DUP']:
            start_region = int(message[1])
            end_region = int(message[1]) + int(message[2])
        else:  # BND
            start_region = max(0, int(message[1]) - 1000)
            end_region = int(message[1]) + 1000
        
        count_region = 0
        for read in bamfile.fetch(contig, start_region, end_region):
            if read.is_supplementary or read.is_secondary or read.is_unmapped:
                continue
            count_region += 1
        
        error_data = float(error_rate1)
        
        if svtype == 'INS':
            min_support = calculate_min_support_ins(globle_coverage, count_region, 0.5, 0.7, 0.3, error_data)
        elif svtype == 'DEL':
            min_support = calculate_min_support_del(globle_coverage, count_region, 0.5, 0.65, 0.35, error_data)
        elif svtype == 'INV':
            min_support = calculate_min_support_inv(globle_coverage, count_region, 0.6, 0.7, 0.3, error_data)
        elif svtype == 'DUP':
            min_support = calculate_min_support_dup(globle_coverage, count_region, 0.5, 0.7, 0.3, error_data)
        else:
            min_support = calculate_min_support_ins(globle_coverage, count_region, 0.5, 0.65, 0.35, error_data)
        
        if svtype == 'INS':
            adjusted_support = _adjust_min_support(
                min_support,
                support_scale=ins_support_scale,
                support_cap=support_cap,
            )
            if int(message[2]) <= int(ins_short_len):
                adjusted_support += max(0, int(ins_short_support_bonus))
            if int(ins_long_len) > 0 and int(message[2]) >= int(ins_long_len):
                adjusted_support += int(ins_long_support_delta)
            if int(ins_min_support_floor) > 0:
                adjusted_support = max(adjusted_support, int(ins_min_support_floor))
        elif svtype == 'INV':
            adjusted_support = _adjust_min_support(
                min_support,
                support_scale=inv_support_scale,
                support_cap=support_cap,
            )
            if int(message[2]) <= int(inv_short_len):
                adjusted_support += max(0, int(inv_short_support_bonus))
            if int(inv_min_support_floor) > 0:
                adjusted_support = max(adjusted_support, int(inv_min_support_floor))
        else:
            adjusted_support = _adjust_min_support(
                min_support,
                support_scale=support_scale,
                support_cap=support_cap,
            )

        message.append(max(1, int(adjusted_support)))
    
    bamfile.close()
    return data


def genotype_multi(candidates: List, bamfile: str, error_rate_list: float, geno_method: str = 'vote') -> List:
    """Calculate genotypes for SV candidates (multi-threaded version)"""
    bam = pysam.AlignmentFile(bamfile, 'rb')
    min_mapq = 20
    geno_method = (geno_method or 'vote').lower()
    
    for candidate in candidates:
        if candidate[6] == 'BND':
            min_support = int(candidate[-1])
            sv_type = candidate[6]
            reads_supporting_variant = candidate[-2]
        else:
            min_support = int(candidate[-1])
            sv_type = candidate[4]
            reads_supporting_variant = candidate[-3]
        
        if len(reads_supporting_variant) < min_support:
            continue
        
        # Determine error rate and search boundaries
        if sv_type == "INS":
            max_bias = 1000
            error_rate = 0.01 if min_support <= 3 else 0.05
            contig, start, end = candidate[0], int(candidate[1]), int(candidate[1]) + 1
        elif sv_type in ['DEL', 'INV', 'DUP']:
            if min_support <= 2:
                error_rate = 0.00
            elif min_support <= 4:
                error_rate = 0.01
            else:
                error_rate = 0.05
            max_bias = 2000
            contig, start, end = candidate[0], int(candidate[1]), int(candidate[1]) + int(candidate[2])
        else:  # BND
            up_bound = threshold_ref_count(len(reads_supporting_variant))
            error_rate = 0.01 if min_support <= 2 else 0.05
            max_bias = 1000
            contig, start = candidate[0], int(candidate[1])
            end = start + 1
        
        contig_length = bam.get_reference_length(contig)
        alignment_it = bam.fetch(contig=contig, start=max(0, start - max_bias), 
                                  stop=min(contig_length, end + max_bias))
        
        aln_no = 0
        reads_supporting_reference = set()
        
        while aln_no < 200:
            try:
                current_alignment = next(alignment_it)
            except StopIteration:
                break
            
            if current_alignment.query_name in reads_supporting_variant:
                continue
            if (current_alignment.is_unmapped or current_alignment.is_secondary or 
                current_alignment.mapping_quality < min_mapq or current_alignment.is_supplementary):
                continue
            
            aln_no += 1
            
            if sv_type in ["DEL", "INV", "DUP"]:
                minimum_overlap = min((end - start) / 2, 2000)
                if ((current_alignment.reference_start < start and 
                     current_alignment.reference_end > start + minimum_overlap) or
                    (current_alignment.reference_start < end - minimum_overlap and 
                     current_alignment.reference_end > end)):
                    reads_supporting_reference.add(current_alignment.query_name)
            else:
                if (current_alignment.reference_start < (start - max_bias) and 
                    current_alignment.reference_end > (end + max_bias)):
                    reads_supporting_reference.add(current_alignment.query_name)
                
                if sv_type == 'BND' and len(reads_supporting_reference) >= up_bound:
                    break
        
        # Calculate genotype
        total_reads = len(reads_supporting_reference) + len(reads_supporting_variant)
        if total_reads < min_support:
            geno = './.'
        else:
            if geno_method == 'loggl':
                geno = genotype_by_log_gl(
                    ref_depth=len(reads_supporting_reference),
                    alt_depth=len(reads_supporting_variant),
                    error_rate=error_rate,
                    prior_AA=1 / 3,
                    prior_AB=1 / 3,
                    prior_BB=1 / 3,
                    bias_factor=0.5,
                )
            else:
                geno = combined_genotype_voting(
                    len(reads_supporting_reference),
                    len(reads_supporting_variant),
                    error_rate, 1 / 3, 1 / 3, 1 / 3
                )
        
        if sv_type == 'BND':
            candidate[-3] = geno
        else:
            candidate[-4] = geno
    
    bam.close()
    return candidates


# ============================================================================
# Main Clustering Function
# ============================================================================

def cluster_by_predict(
    bamfilepath: str,
    data_path: str,
    testpath: str,
    outputpath: str,
    contigg: List,
    threads_numm: int = 15,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    bp_refine: bool = False,
    bp_refine_window: int = 200,
    bp_refine_min_evidence: int = 2,
    bp_refine_max_shift: int = 500,
    bp_refine_types: str = 'DEL',
    bp_refine_ins_min_evidence: Optional[int] = None,
    bp_refine_ins_max_shift: Optional[int] = None,
    bp_refine_ins_min_len_ratio: Optional[float] = None,
    bp_refine_ins_max_len_ratio: Optional[float] = None,
    geno_method: str = 'loggl',
    max_dup_inv_len: int = DEFAULT_MAX_DUP_INV_LEN,
    support_scale: float = 1.0,
    support_cap: int = 0,
    ins_support_scale: float = 1.0,
    ins_min_support_floor: int = 0,
    ins_short_len: int = 300,
    ins_short_support_bonus: int = 0,
    ins_long_len: int = 0,
    ins_long_support_delta: int = 0,
    inv_support_scale: float = 1.0,
    inv_min_support_floor: int = 0,
    inv_short_len: int = 3000,
    inv_short_support_bonus: int = 0,
    ins_eps_bands: Optional[str] = None,
    inv_eps_bands: Optional[str] = None,
):
    """
    Main function to cluster predictions and generate VCF output
    
    Args:
        bamfilepath: Path to BAM file
        data_path: Path to feature data
        testpath: Path to predictions
        outputpath: Path for output VCF files
        contigg: List of contigs to process
        threads_numm: Number of threads
        decision_threshold: Probability cutoff used to convert window scores into SV candidate regions
        geno_method: Genotype strategy ('vote' or 'loggl')
        max_dup_inv_len: Maximum allowed length for DUP/INV candidates (bp), 0 disables cap
        support_scale: Multiplicative calibration applied to inferred minimum support
        support_cap: Optional hard upper cap applied to inferred minimum support (0 disables cap)
        bp_refine: Whether to enable breakpoint refinement
        bp_refine_window: Search window around predicted breakpoints
        bp_refine_min_evidence: Minimum evidence reads to apply refinement
        bp_refine_max_shift: Maximum allowed breakpoint shift
        bp_refine_types: Comma-separated SV types to refine
        bp_refine_ins_min_evidence: INS-specific minimum evidence reads (optional)
        bp_refine_ins_max_shift: INS-specific max allowed breakpoint shift (optional)
        bp_refine_ins_min_len_ratio: INS-specific minimum refined/original length ratio (optional)
        bp_refine_ins_max_len_ratio: INS-specific maximum refined/original length ratio (optional)
    """
    threads_num = max(1, int(threads_numm))
    decision_threshold = float(np.clip(decision_threshold, 0.0, 1.0))
    geno_method = (geno_method or 'vote').lower()
    max_dup_inv_len = max(0, int(max_dup_inv_len))
    support_scale = float(support_scale)
    support_cap = max(0, int(support_cap))
    ins_eps_bands_parsed = _parse_eps_band_spec(ins_eps_bands)
    inv_eps_bands_parsed = _parse_eps_band_spec(inv_eps_bands)
    if geno_method not in {'vote', 'loggl'}:
        logger.warning(f"Unknown geno_method={geno_method}, fallback to 'vote'")
        geno_method = 'vote'
    logger.info("Using decision threshold: %.3f", decision_threshold)
    if max_dup_inv_len > 0:
        logger.info(f"Applying DUP/INV length cap: {max_dup_inv_len} bp")
    else:
        logger.info("DUP/INV length cap disabled")
    logger.info("Support calibration: scale=%.3f cap=%s", support_scale, support_cap if support_cap > 0 else "disabled")
    logger.info(
        "INS support calibration: scale=%.3f floor=%s short_len=%d short_bonus=%d long_len=%d long_delta=%d",
        ins_support_scale,
        ins_min_support_floor if int(ins_min_support_floor) > 0 else "disabled",
        int(ins_short_len),
        int(ins_short_support_bonus),
        int(ins_long_len),
        int(ins_long_support_delta),
    )
    logger.info(
        "INV support calibration: scale=%.3f floor=%s short_len=%d short_bonus=%d",
        inv_support_scale,
        inv_min_support_floor if int(inv_min_support_floor) > 0 else "disabled",
        int(inv_short_len),
        int(inv_short_support_bonus),
    )
    logger.info("INS length-aware eps bands: %s", _format_eps_band_spec(ins_eps_bands_parsed or _length_aware_eps_bands('INS', 1.0)))
    logger.info("INV length-aware eps bands: %s", _format_eps_band_spec(inv_eps_bands_parsed or _length_aware_eps_bands('INV', 1.0)))
    filename = os.path.basename(bamfilepath)
    name_without_extension = os.path.splitext(filename)[0]
    directory_path = os.path.join(outputpath, name_without_extension)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    output_del_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_del.vcf"
    output_ins_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_ins.vcf"
    output_inv_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_inv.vcf"
    output_dup_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_dup.vcf"
    output_bnd_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_tra.vcf"
    output_all_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_all.vcf"

    contigg, contig2length, prediction_jobs = collect_prediction_jobs(
        bamfilepath=bamfilepath,
        data_path=data_path,
        testpath=testpath,
        contigg=contigg,
    )

    ready_jobs, missing_jobs, empty_store_count = summarize_prediction_jobs(prediction_jobs)
    if missing_jobs:
        logger.info(
            "Skipping %d regions without reusable prediction files under %s",
            len(missing_jobs),
            testpath,
        )
    if empty_store_count > 0:
        logger.info("Ignoring %d empty feature stores during clustering", empty_store_count)

    jobs_by_contig: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job in ready_jobs:
        jobs_by_contig[str(job['chr_name'])].append(job)

    with Pool(threads_num) as thread_pool:
        bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=1)
        try:
            # Calculate global coverage
            coverage_task = [(bamfilepath, chr_name, contig2length[chr_name]) for chr_name in contigg]
            global_coverage_all = thread_pool.starmap(average_read_coverage, coverage_task)
            alllength = [contig2length[chr_name] for chr_name in contigg]
            global_coverage = math.ceil(sum(global_coverage_all) / sum(alllength)) if alllength else 0

            # Load prediction data
            x_train_data_all = []
            for chr_name in contigg:
                contig_jobs = jobs_by_contig.get(str(chr_name), [])
                logger.info(f"Processing {chr_name} ({len(contig_jobs)} segments)")
                for job in contig_jobs:
                    x_train_data_all.append([
                        job['predict_path'],
                        str(job['data_store_path']),
                        int(job['start']),
                        int(job['end']),
                        str(job['chr_name']),
                    ])

            # Process predictions
            tasks = [
                (
                    predict_path,
                    data_store_path,
                    start,
                    end,
                    chr_name,
                    bamfilepath,
                    decision_threshold,
                    max_dup_inv_len,
                )
                for predict_path, data_store_path, start, end, chr_name in x_train_data_all
            ]

            parsing_results = thread_pool.starmap(load_all_data, tasks)

            # Analyze candidates
            tasks_candidate = [
                (parsing_results, chrname, max_dup_inv_len, ins_eps_bands_parsed, inv_eps_bands_parsed)
                for chrname in contigg
            ]
            resultlist_begin = thread_pool.starmap(analysis_candidate, tasks_candidate)

            # Extract error rates
            error_all = {}
            for i in resultlist_begin:
                try:
                    error_all[i[0][0][0]] = i[1]
                except IndexError:
                    continue

            # Calculate support and genotype
            logger.info("Calculating support reads...")
            tasks_support = []
            dropped_long_support = 0
            for chrname in contigg:
                resultcontig = []
                for resultl in resultlist_begin:
                    for result in resultl[0]:
                        if result[0] == chrname:
                            # BND records have different field layout; keep them unchanged.
                            if len(result) > 6 and str(result[6]) == 'BND':
                                resultcontig.append(result)
                                continue
                            svtype = str(result[4])
                            try:
                                svlen = abs(int(float(result[2])))
                            except (TypeError, ValueError):
                                svlen = 0
                            # Guard against pathological long events that can dominate BAM scanning time.
                            if max_dup_inv_len > 0 and svtype in {'DEL', 'INV', 'DUP'} and svlen > max_dup_inv_len:
                                dropped_long_support += 1
                                continue
                            resultcontig.append(result)
                if not resultcontig:
                    continue
                error_rate = error_all.get(chrname, [0])
                chunk_size = max(1, math.ceil(len(resultcontig) / max(1, threads_num)))
                for idx in range(0, len(resultcontig), chunk_size):
                    tasks_support.append((
                        resultcontig[idx:idx + chunk_size],
                        bamfilepath,
                        global_coverage,
                        error_rate,
                        support_scale,
                        support_cap,
                        ins_support_scale,
                        ins_min_support_floor,
                        ins_short_len,
                        ins_short_support_bonus,
                        ins_long_len,
                        ins_long_support_delta,
                        inv_support_scale,
                        inv_min_support_floor,
                        inv_short_len,
                        inv_short_support_bonus,
                    ))
            if dropped_long_support > 0:
                logger.info(
                    "Dropped %d DEL/INV/DUP candidates longer than %d bp before support counting",
                    dropped_long_support,
                    max_dup_inv_len,
                )
            logger.info("Support tasks split into %d chunks", len(tasks_support))
            resultlist1 = thread_pool.starmap(support_read_calculate_multi, tasks_support)

            logger.info("Calculating genotypes...")
            tasks_geno = []
            for chrname in contigg:
                resultcontig = []
                for resultl in resultlist1:
                    for result in resultl:
                        if result[0] == chrname:
                            resultcontig.append(result)
                if not resultcontig:
                    continue
                error_rate = error_all.get(chrname, [0])
                chunk_size = max(1, math.ceil(len(resultcontig) / max(1, threads_num)))
                for idx in range(0, len(resultcontig), chunk_size):
                    tasks_geno.append((resultcontig[idx:idx + chunk_size], bamfilepath, error_rate, geno_method))
            logger.info("Genotype tasks split into %d chunks", len(tasks_geno))
            resultlist = thread_pool.starmap(genotype_multi, tasks_geno)

            # Collect results
            all_result = []
            BND_result = []

            for read in resultlist:
                for read1 in read:
                    if str(read1[6]) != 'BND':
                        if int(read1[3]) >= int(read1[8]) and int(float(read1[2])) >= 40:
                            all_result.append([
                                str(read1[0]), int(float(read1[1])), int(float(read1[2])),
                                int(float(read1[3])), str(read1[4]), read1[-4], read1[-2], read1[-1]
                            ])
                    elif str(read1[6]) == 'BND':
                        if int(read1[7]) >= int(read1[10]) and str(read1[0]) in contigg and str(read1[3]) in contigg:
                            BND_result.append([read1])

            if bp_refine:
                logger.info(
                    "Running breakpoint refinement (types=%s, window=%dbp, min_evidence=%d, max_shift=%dbp)",
                    bp_refine_types,
                    int(bp_refine_window),
                    int(bp_refine_min_evidence),
                    int(bp_refine_max_shift),
                )
                logger.info(
                    "INS refine overrides (min_evidence=%s, max_shift=%s, min_len_ratio=%s, max_len_ratio=%s)",
                    str(bp_refine_ins_min_evidence),
                    str(bp_refine_ins_max_shift),
                    str(bp_refine_ins_min_len_ratio),
                    str(bp_refine_ins_max_len_ratio),
                )
                all_result, refine_summary = breakpoint_refine(
                    records=all_result,
                    bamfile=bamfile,
                    enabled=True,
                    refine_types=bp_refine_types,
                    window_bp=bp_refine_window,
                    min_evidence=bp_refine_min_evidence,
                    max_shift=bp_refine_max_shift,
                    ins_min_evidence=bp_refine_ins_min_evidence,
                    ins_max_shift=bp_refine_ins_max_shift,
                    ins_min_len_ratio=bp_refine_ins_min_len_ratio,
                    ins_max_len_ratio=bp_refine_ins_max_len_ratio,
                )
                changed_count = sum(int(row[8]) for row in refine_summary)
                logger.info("Breakpoint refinement updated %d/%d records", changed_count, len(refine_summary))
                summary_path = f"{outputpath}/{name_without_extension}/{name_without_extension}_bp_refine.tsv"
                write_refine_summary(refine_summary, summary_path)
                if refine_summary:
                    logger.info(f"Refinement summary written: {summary_path}")

            # Separate by SV type
            inss = [r for r in all_result if r[4] == 'INS']
            delll = [r for r in all_result if r[4] == 'DEL']
            dup = [r for r in all_result if r[4] == 'DUP']
            inv = [r for r in all_result if r[4] == 'INV']

            logger.info(f"Results - DEL: {len(delll)}, INS: {len(inss)}, INV: {len(inv)}, DUP: {len(dup)}, BND: {len(BND_result)}")

            # Generate VCF files
            logger.info("Generating VCF files...")
            try:
                generate_vcf(all_result, contig2length, output_all_path, BND_result)
                generate_vcf(inss, contig2length, output_ins_path)
                generate_vcf(delll, contig2length, output_del_path)
                generate_vcf(dup, contig2length, output_dup_path)
                generate_vcf(inv, contig2length, output_inv_path)
                generate_vcf_bnd(inss, contig2length, output_bnd_path, BND_result)
            except Exception:
                debug_path = f"{output_all_path}.debug.pkl"
                try:
                    _dump_vcf_debug_snapshot(debug_path, all_result, BND_result, contig2length)
                    logger.exception("VCF generation failed; wrote debug snapshot to %s", debug_path)
                except Exception:
                    logger.exception("VCF generation failed and debug snapshot writing also failed")
                raise
        finally:
            bamfile.close()

    logger.info("Clustering completed")


if __name__ == '__main__':
    print("CMSV Detection Module (PyTorch Version)")
