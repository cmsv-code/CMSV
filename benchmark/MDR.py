import argparse
import logging
import os
from bisect import bisect_left, bisect_right


NON_BND_TYPES = ["DEL", "INS", "DUP", "INV"]
REPORT_TYPES = ["DEL", "INS", "DUP", "INV", "BND"]
INTERVAL_BIN = 1_000_000


def parse_info(seq):
    info = {"SVLEN": 0, "END": 0, "SVTYPE": "", "CHR2": ""}
    for kv in seq.split(";"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k in ["SVLEN", "END"]:
            try:
                info[k] = abs(int(float(v)))
            except Exception:
                pass
        elif k == "SVTYPE":
            info[k] = v[:3]
        elif k == "CHR2":
            info[k] = v
    return info


def parse_bnd_alt(alt):
    chr2 = ""
    pos2 = 0
    form = ""
    try:
        if alt.startswith("]"):
            form = "]]N"
            chr2 = alt.split(":")[0][1:]
            pos2 = int(alt.split(":")[1].split("]")[0])
        elif alt.startswith("["):
            form = "[[N"
            chr2 = alt.split(":")[0][1:]
            pos2 = int(alt.split(":")[1].split("[")[0])
        else:
            if len(alt) > 1 and alt[1] == "]":
                form = "N]]"
                chr2 = alt.split(":")[0][2:]
                pos2 = int(alt.split(":")[1].split("]")[0])
            else:
                form = "N[["
                chr2 = alt.split(":")[0][2:]
                pos2 = int(alt.split(":")[1].split("[")[0])
    except Exception:
        pass
    return chr2, pos2, form


def load_records(path):
    records = []
    with open(path, "r") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            seq = line.rstrip("\n").split("\t")
            if len(seq) < 8:
                continue
            chrom = seq[0]
            try:
                pos = int(seq[1])
            except Exception:
                continue
            info = parse_info(seq[7])
            svtype = info["SVTYPE"]
            if svtype == "BND":
                chr2, pos2, form = parse_bnd_alt(seq[4])
                if info["END"] == 0:
                    info["END"] = pos2
                if not info["CHR2"]:
                    info["CHR2"] = chr2
                records.append((chrom, pos, info["CHR2"], info["END"], form))
            elif svtype in NON_BND_TYPES:
                svlen = info["SVLEN"]
                if svlen == 0 and info["END"] >= pos:
                    svlen = info["END"] - pos + 1
                records.append((chrom, pos, info["END"], max(svlen, 0)))
    return records


def resolve_sample_base(sample_dir):
    direct_candidates = [p for p in os.listdir(sample_dir) if p.endswith("_del.vcf")]
    if direct_candidates:
        return os.path.join(sample_dir, direct_candidates[0][:-8])

    subdirs = [p for p in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, p))]
    if len(subdirs) != 1:
        raise RuntimeError(f"Cannot resolve sample base from {sample_dir}")
    return os.path.join(sample_dir, subdirs[0])


def load_sample_dir(sample_dir):
    base = resolve_sample_base(sample_dir)
    return {
        "DEL": load_records(base + "_del.vcf"),
        "INS": load_records(base + "_ins.vcf"),
        "DUP": load_records(base + "_dup.vcf"),
        "INV": load_records(base + "_inv.vcf"),
        "BND": load_records(base + "_tra.vcf"),
    }


def merge_parents(father, mother):
    merged = {}
    for svtype in REPORT_TYPES:
        merged[svtype] = father.get(svtype, []) + mother.get(svtype, [])
    return merged


def build_ins_index(records, offect):
    idx = {}
    bucket_size = max(offect, 1)
    for chrom, pos, end, svlen in records:
        idx.setdefault(chrom, {}).setdefault(pos // bucket_size, []).append((pos, svlen))
    return idx


def build_interval_index(records, offect):
    idx = {}
    for chrom, pos, end, svlen in records:
        start_bin = (pos - offect) // INTERVAL_BIN
        end_bin = (end + offect) // INTERVAL_BIN
        chrom_map = idx.setdefault(chrom, {})
        for bucket in range(start_bin, end_bin + 1):
            chrom_map.setdefault(bucket, []).append((pos, end, svlen))
    return idx


def build_bnd_index(records, offect):
    idx = {}
    bucket_size = max(offect, 1)
    for chrom, pos, chr2, end, form in records:
        key = (chrom, chr2, form, pos // bucket_size, end // bucket_size)
        idx.setdefault(key, []).append((pos, end))
    return idx


def len_ok(len_a, len_b, bias):
    maxlen = max(len_a, len_b, 1)
    minlen = min(len_a, len_b)
    return float(minlen) / float(maxlen) >= bias


def count_matches_ins(child_records, parent_index, bias, offect):
    matched = 0
    bucket_size = max(offect, 1)
    for chrom, pos, end, svlen in child_records:
        chrom_map = parent_index.get(chrom, {})
        found = False
        left = (pos - offect) // bucket_size
        right = (pos + offect) // bucket_size
        for bucket in range(left, right + 1):
            for parent_pos, parent_len in chrom_map.get(bucket, []):
                if abs(parent_pos - pos) <= offect and len_ok(parent_len, svlen, bias):
                    matched = matched + 1
                    found = True
                    break
            if found:
                break
    return matched


def count_matches_interval(child_records, parent_index, bias, offect):
    matched = 0
    for chrom, pos, end, svlen in child_records:
        found = False
        chrom_map = parent_index.get(chrom, {})
        left = (pos - offect) // INTERVAL_BIN
        right = (end + offect) // INTERVAL_BIN
        for bucket in range(left, right + 1):
            for parent_pos, parent_end, parent_len in chrom_map.get(bucket, []):
                if max(parent_pos - offect, pos) <= min(parent_end + offect, end):
                    if len_ok(parent_len, svlen, bias):
                        matched = matched + 1
                        found = True
                        break
            if found:
                break
    return matched


def count_matches_bnd(child_records, parent_index, offect):
    matched = 0
    bucket_size = max(offect, 1)
    for chrom, pos, chr2, end, form in child_records:
        found = False
        for b1 in range((pos - offect) // bucket_size, (pos + offect) // bucket_size + 1):
            for b2 in range((end - offect) // bucket_size, (end + offect) // bucket_size + 1):
                key = (chrom, chr2, form, b1, b2)
                for parent_pos, parent_end in parent_index.get(key, []):
                    if abs(parent_pos - pos) <= offect and abs(parent_end - end) <= offect:
                        matched = matched + 1
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return matched


def evaluate_platform(father_dir, mother_dir, child_dir, bias, offect):
    father = load_sample_dir(father_dir)
    mother = load_sample_dir(mother_dir)
    child = load_sample_dir(child_dir)
    parents = merge_parents(father, mother)

    interval_indexes = {
        "DEL": build_interval_index(parents["DEL"], offect),
        "DUP": build_interval_index(parents["DUP"], offect),
        "INV": build_interval_index(parents["INV"], offect),
    }
    ins_index = build_ins_index(parents["INS"], offect)
    bnd_index = build_bnd_index(parents["BND"], offect)

    result = {}
    total_records = 0
    total_matched = 0
    for svtype in REPORT_TYPES:
        child_records = child[svtype]
        total = len(child_records)
        if svtype == "INS":
            matched = count_matches_ins(child_records, ins_index, bias, offect)
        elif svtype == "BND":
            matched = count_matches_bnd(child_records, bnd_index, offect)
        else:
            matched = count_matches_interval(child_records, interval_indexes[svtype], bias, offect)
        inconsistent = total - matched
        mdr = 0.0 if total == 0 else 100.0 * inconsistent / total
        result[svtype] = {"TOTAL": total, "MATCHED": matched, "INCONSISTENT": inconsistent, "MDR": mdr}
        total_records += total
        total_matched += matched

    total_inconsistent = total_records - total_matched
    result["ALL"] = {
        "TOTAL": total_records,
        "MATCHED": total_matched,
        "INCONSISTENT": total_inconsistent,
        "MDR": 0.0 if total_records == 0 else 100.0 * total_inconsistent / total_records,
    }
    return result


def print_result(label, result):
    print(f"[{label}]")
    print(f"{'SV':<5} {'TOTAL':>9} {'INCONSISTENT':>12} {'MDR(%)':>9}")
    for svtype in ["DEL", "INS", "DUP", "INV", "BND", "ALL"]:
        display = "Total" if svtype == "ALL" else svtype
        item = result[svtype]
        print(f"{display:<5} {item['TOTAL']:>9d} {item['INCONSISTENT']:>12d} {item['MDR']:>9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("father_dir")
    parser.add_argument("mother_dir")
    parser.add_argument("child_dir")
    parser.add_argument("-b", "--bias", type=float, default=0.7)
    parser.add_argument("-o", "--offect", type=int, default=1000)
    parser.add_argument("--label", default="trio")
    args = parser.parse_args()

    result = evaluate_platform(args.father_dir, args.mother_dir, args.child_dir, args.bias, args.offect)
    print_result(args.label, result)


if __name__ == "__main__":
    main()
