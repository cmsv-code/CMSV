import gzip
from pathlib import Path


ROOT = Path("/mnt/HDD8TB/zqchengsong/trio_eval_run_merged")
OUT = Path("/data/zqchengsong/mambsv/variantcaller_clean_20260314/trio_eval_run/mier_hg567_newscript_missing_parent_dot")
BIAS = 0.7
OFFECT = 1000
INTERVAL_BIN = 1_000_000

PLATFORMS = {
    "ccs": {
        "father": ROOT / "ccs/hg006_ccs_30x/vcf/HG006.GRCh38.haplotagged_w_10x/HG006.GRCh38.haplotagged_w_10x_all.sorted.vcf.gz",
        "mother": ROOT / "ccs/hg007_ccs_30x/vcf/HG007.GRCh38.haplotagged_w_10x/HG007.GRCh38.haplotagged_w_10x_all.sorted.vcf.gz",
        "child": ROOT / "ccs/hg005_ccs_30x/vcf/HG005.GRCh38.haplotagged/HG005.GRCh38.haplotagged_all.sorted.vcf.gz",
    },
    "clr": {
        "father": ROOT / "clr/hg006_clr_30x/vcf/HG006_PacBio_GRCh38.30x/HG006_PacBio_GRCh38.30x_all.sorted.vcf.gz",
        "mother": ROOT / "clr/hg007_clr_30x/vcf/HG007_PacBio_GRCh38.30x/HG007_PacBio_GRCh38.30x_all.sorted.vcf.gz",
        "child": ROOT / "clr/hg005_clr_30x/vcf/HG005_PacBio_GRCh38.30x/HG005_PacBio_GRCh38.30x_all.sorted.vcf.gz",
    },
    "ont": {
        "father": ROOT / "ont/hg006_ont_30x/vcf/HG006_GRCh38_ONT-UL_UCSC_20200109.phased.30x/HG006_GRCh38_ONT-UL_UCSC_20200109.phased.30x_all.sorted.vcf.gz",
        "mother": ROOT / "ont/hg007_ont_30x/vcf/HG007_GRCh38_ONT-UL_UCSC_20200109.phased.30x/HG007_GRCh38_ONT-UL_UCSC_20200109.phased.30x_all.sorted.vcf.gz",
        "child": ROOT / "ont/hg005_ont_30x/vcf/HG005_GRCh38_ONT-UL_UCSC_20200109.phased.30x/HG005_GRCh38_ONT-UL_UCSC_20200109.phased.30x_all.sorted.vcf.gz",
    },
}


def open_vcf(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def parse_info(text):
    info = {"SVTYPE": "", "END": 0, "SVLEN": 0, "CHR2": ""}
    for item in text.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key == "SVTYPE":
            svtype = value.upper()
            info[key] = "BND" if svtype == "TRA" else svtype[:3]
        elif key in ("END", "SVLEN"):
            try:
                info[key] = abs(int(float(value)))
            except Exception:
                pass
        elif key == "CHR2":
            info[key] = value
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


def parse_gt(fmt, sample):
    keys = fmt.split(":")
    vals = sample.split(":")
    if "GT" not in keys:
        return "1/1"
    idx = keys.index("GT")
    gt = vals[idx] if idx < len(vals) else "1/1"
    if gt in (".", "./.", ".|.", ""):
        return "1/1"
    return gt.replace("|", "/")


def load_parent_records(path):
    records = {sv: [] for sv in ["DEL", "INS", "DUP", "INV", "BND"]}
    with open_vcf(path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            chrom = fields[0]
            try:
                pos = int(fields[1])
            except Exception:
                continue
            info = parse_info(fields[7])
            svtype = info["SVTYPE"]
            if svtype not in records:
                continue
            gt = parse_gt(fields[8], fields[9])
            if svtype == "BND":
                chr2, pos2, form = parse_bnd_alt(fields[4])
                end = info["END"] or pos2
                chr2 = info["CHR2"] or chr2
                records[svtype].append((chrom, pos, chr2, end, form, gt))
            else:
                svlen = info["SVLEN"]
                end = info["END"]
                if svlen == 0 and end >= pos:
                    svlen = end - pos + 1
                records[svtype].append((chrom, pos, end, max(svlen, 0), gt))
    return records


def len_ok(a, b):
    return min(a, b) / max(a, b, 1) >= BIAS


def build_ins_index(records):
    idx = {}
    bucket = max(OFFECT, 1)
    for chrom, pos, end, svlen, gt in records:
        idx.setdefault(chrom, {}).setdefault(pos // bucket, []).append((pos, svlen, gt))
    return idx


def build_interval_index(records):
    idx = {}
    for chrom, pos, end, svlen, gt in records:
        start_bin = (pos - OFFECT) // INTERVAL_BIN
        end_bin = (end + OFFECT) // INTERVAL_BIN
        for bucket in range(start_bin, end_bin + 1):
            idx.setdefault(chrom, {}).setdefault(bucket, []).append((pos, end, svlen, gt))
    return idx


def build_bnd_index(records):
    idx = {}
    bucket = max(OFFECT, 1)
    for chrom, pos, chr2, end, form, gt in records:
        key = (chrom, chr2, form, pos // bucket, end // bucket)
        idx.setdefault(key, []).append((pos, end, gt))
    return idx


def build_parent_indexes(records):
    return {
        "DEL": build_interval_index(records["DEL"]),
        "DUP": build_interval_index(records["DUP"]),
        "INV": build_interval_index(records["INV"]),
        "INS": build_ins_index(records["INS"]),
        "BND": build_bnd_index(records["BND"]),
    }


def find_parent_gt(svtype, child, indexes):
    if svtype == "INS":
        chrom, pos, end, svlen = child
        chrom_map = indexes["INS"].get(chrom, {})
        bucket = max(OFFECT, 1)
        for b in range((pos - OFFECT) // bucket, (pos + OFFECT) // bucket + 1):
            for ppos, plen, gt in chrom_map.get(b, []):
                if abs(ppos - pos) <= OFFECT and len_ok(plen, svlen):
                    return gt
        return "./."
    if svtype == "BND":
        chrom, pos, chr2, end, form = child
        bucket = max(OFFECT, 1)
        for b1 in range((pos - OFFECT) // bucket, (pos + OFFECT) // bucket + 1):
            for b2 in range((end - OFFECT) // bucket, (end + OFFECT) // bucket + 1):
                key = (chrom, chr2, form, b1, b2)
                for ppos, pend, gt in indexes["BND"].get(key, []):
                    if abs(ppos - pos) <= OFFECT and abs(pend - end) <= OFFECT:
                        return gt
        return "./."

    chrom, pos, end, svlen = child
    chrom_map = indexes[svtype].get(chrom, {})
    for b in range((pos - OFFECT) // INTERVAL_BIN, (end + OFFECT) // INTERVAL_BIN + 1):
        for ppos, pend, plen, gt in chrom_map.get(b, []):
            if max(ppos - OFFECT, pos) <= min(pend + OFFECT, end) and len_ok(plen, svlen):
                return gt
    return "./."


def child_match_tuple(fields, info):
    chrom = fields[0]
    pos = int(fields[1])
    svtype = info["SVTYPE"]
    if svtype == "BND":
        chr2, pos2, form = parse_bnd_alt(fields[4])
        return (chrom, pos, info["CHR2"] or chr2, info["END"] or pos2, form)
    svlen = info["SVLEN"]
    end = info["END"]
    if svlen == 0 and end >= pos:
        svlen = end - pos + 1
    return (chrom, pos, end, max(svlen, 0))


def make_trio_vcf(platform, cfg):
    father_idx = build_parent_indexes(load_parent_records(cfg["father"]))
    mother_idx = build_parent_indexes(load_parent_records(cfg["mother"]))
    out_vcf = OUT / "trio_vcfs" / f"hg567_{platform}.trio.vcf"
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    counts = {}

    with open_vcf(cfg["child"]) as inp, out_vcf.open("w") as out:
        for line in inp:
            if line.startswith("##"):
                out.write(line)
                continue
            if line.startswith("#CHROM"):
                out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG006_father\tHG007_mother\tHG005_child\n")
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            info = parse_info(fields[7])
            svtype = info["SVTYPE"]
            if svtype not in ["DEL", "INS", "DUP", "INV", "BND"]:
                continue
            child_tuple = child_match_tuple(fields, info)
            father_gt = find_parent_gt(svtype, child_tuple, father_idx)
            mother_gt = find_parent_gt(svtype, child_tuple, mother_idx)
            child_gt = parse_gt(fields[8], fields[9])
            fields[8] = "GT"
            fields[9:] = [father_gt, mother_gt, child_gt]
            out.write("\t".join(fields) + "\n")
            counts[svtype] = counts.get(svtype, 0) + 1
    return out_vcf, counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "trio_vcf_manifest.tsv"
    with manifest.open("w") as mf:
        mf.write("platform\ttrio_vcf\tDEL\tINS\tDUP\tINV\tBND\n")
        for platform, cfg in PLATFORMS.items():
            trio_vcf, counts = make_trio_vcf(platform, cfg)
            mf.write(
                f"{platform}\t{trio_vcf}\t"
                f"{counts.get('DEL', 0)}\t{counts.get('INS', 0)}\t{counts.get('DUP', 0)}\t"
                f"{counts.get('INV', 0)}\t{counts.get('BND', 0)}\n"
            )
            print(f"[done] {platform} {trio_vcf} {counts}", flush=True)
    print(manifest)


if __name__ == "__main__":
    main()
