#!/usr/bin/env python3
import argparse, json, re
import pandas as pd
from pathlib import Path

def count_overlapping(text: str, sub: str) -> int:
    if not sub: 
        return 0
    return sum(1 for _ in re.finditer(rf"(?={re.escape(sub)})", text))

def make_tuples(text: str, spans):
    tuples = []
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    for sp in spans:
        try:
            start, end = int(sp["start"]), int(sp["end"])
            label = str(sp.get("label", ""))
        except Exception:
            continue
        if start < 0 or end < 0 or end > len(text) or end < start:
            continue
        sub = text[start:end]
        occ = count_overlapping(text, sub)
        tuples.append((sub, occ, label))
    return tuples

def robust_read(path: Path):
    """
    Read a 'broken' CSV where the last column (pii_spans) is always JSON.
    Strategy:
      - read the whole file line by line
      - split only at the *last* comma → everything before = other columns, after = spans
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n")
        colnames = header.split(",")
        rows = []
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # split at last comma
            prefix, sep, spans_str = line.rpartition(",")
            print("prefix:", prefix)
            print("spans_str:", spans_str)
            
            spans = []
            try:
                spans = json.loads(spans_str)
            except Exception:
                pass
            row_parts = prefix.split(",")
            # pad if mismatch
            while len(row_parts) < len(colnames)-1:
                row_parts.append("")
            row = dict(zip(colnames[:-1], row_parts))
            row["pii_spans"] = spans
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    df = robust_read(Path(args.inp))

    # Build tuples
    new_col = []
    for _, row in df.iterrows():
        text = row.get("generated_text", "")
        print("text:", text)
        spans = row.get("pii_spans", [])
        print("spans:", spans)
        tuples = make_tuples(text, spans)
        print(tuples)
        new_col.append(json.dumps(tuples, ensure_ascii=False))
    df["pii_tuples"] = new_col

    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()
