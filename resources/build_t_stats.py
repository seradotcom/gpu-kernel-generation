"""
build_t_stats.py  —  Reconstruct the missing TritonBench-T statistics file.

The TritonBench-T eval scripts (EVAL/eval_T/0_call_acc.py, 1_exe_acc.py) read a
stats file `statis_path` whose entries look like {"description": ..., "file": ...}.
They use it in get_corresponding_files() to map each instruction back to its gold
operator file by checking that the instruction's "Functional Description:" text is a
substring of exactly one entry's "description".

That T stats file is NOT shipped in the repo (only TritonBench_G_v1.json exists), so
this script rebuilds it from the alpaca instructions + the gold operator folder.

Mapping strategy (verified to give a unique 1:1 match for all 166 entries):
  - Primary key: parse the entry-function name from "Wrapper Entry Information:" and
    map it to "<name>.py" in the gold folder (164/166 resolve this way).
  - Two formatting outliers are handled by an explicit override table.
  - "description" is set to the exact functional-description string the eval extracts,
    guaranteeing the substring match resolves to exactly one file.

Run from the TritonBench repo root:
    python build_t_stats.py
"""

import json
import os
import re
import argparse

# Two instructions whose "Wrapper Entry Information" is formatted so the function name
# is not the first token; resolved by their functional description.
OVERRIDES = {
    31: "mean.py",   # "Returns the mean value of each row of the input tensor ..."
    141: "solve.py", # "Computes the solution of a square system of linear equations ..."
}


def entry_name(instruction: str):
    seg = instruction.split("Wrapper Entry Information:")[-1]
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", seg)
    return m.group(1) if m else None


def functional_description(instruction: str) -> str:
    return (
        instruction.split("Functional Description: ")[-1]
        .split("Wrapper Entry Information:")[0]
        .replace("\n", "")
        .strip()
    )


def build(alpaca_path: str, gold_dir: str, out_path: str):
    alpaca = json.load(open(alpaca_path, "r", encoding="utf-8"))
    gold = set(os.listdir(gold_dir))

    stats = []
    for i, entry in enumerate(alpaca):
        instr = entry["instruction"]
        fname = OVERRIDES.get(i)
        if fname is None:
            n = entry_name(instr)
            fname = f"{n}.py" if n else None
        assert fname in gold, f"[idx {i}] could not map to a gold file (got {fname!r})"
        stats.append({"description": functional_description(instr), "file": fname})

    # Verify the eval's matching logic resolves each instruction to exactly one file.
    for i, entry in enumerate(alpaca):
        func = (
            entry["instruction"]
            .split("Functional Description: ")[-1]
            .split("Wrapper Entry Information:")[0]
            .replace("\n", "")
        )
        hits = [s["file"] for s in stats if func in s["description"].replace("\n", "")]
        assert len(hits) == 1, f"[idx {i}] resolves to {len(hits)} files: {hits}"

    json.dump(stats, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(stats)} entries -> {out_path}")
    print(f"Distinct gold files referenced: {len({s['file'] for s in stats})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", default="data/TritonBench_T_simp_alpac_v1.json",
                    help="Path to the simple-instruction alpaca file (or the _comp_ one).")
    ap.add_argument("--gold_dir", default="data/TritonBench_T_v1",
                    help="Folder of gold operator .py files.")
    ap.add_argument("--out", default="data/TritonBench_T_v1.json",
                    help="Output stats file path (must match statis_path in the eval scripts).")
    args = ap.parse_args()
    build(args.alpaca, args.gold_dir, args.out)
