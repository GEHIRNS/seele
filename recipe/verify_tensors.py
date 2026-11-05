"""Check a GGUF against a recipe table: same tensor names, ggml types and data SHA-256 for every tensor.

    pip install gguf==0.19.0
    python verify_tensors.py seele-1.0-3.3B.gguf seele-1.0-3.3B.tensor-types.tsv

Exit code 0 when all tensors match. Exit code 2 when the only differences are in ssm_a tensors: those are F32 values
that the BF16 conversion computes as -exp(A_log), and a conversion on another machine can round a few of them 1 ulp
differently (see BUILD.md). Metadata is not compared.
"""
import hashlib
import sys

import gguf

gguf_path, tsv_path = sys.argv[1], sys.argv[2]
want = {}
for line in open(tsv_path, encoding="utf-8").read().splitlines()[1:]:
    c = line.split("\t")
    want[c[0]] = (c[1], c[6])

reader = gguf.GGUFReader(gguf_path)
bad, ssm_a = [], []
for t in reader.tensors:
    if t.name not in want:
        bad.append(f"{t.name}: not in the recipe table")
        continue
    ttype, digest = want.pop(t.name)
    if t.tensor_type.name != ttype:
        bad.append(f"{t.name}: type {t.tensor_type.name}, expected {ttype}")
    elif hashlib.sha256(t.data.tobytes()).hexdigest() != digest:
        (ssm_a if t.name.endswith(".ssm_a") else bad).append(f"{t.name}: data differs")
bad += [f"{name}: missing from the file" for name in want]

n = len(reader.tensors)
if bad:
    print(f"{len(bad) + len(ssm_a)} problem(s) in {n} tensors:")
    print("\n".join("  " + b for b in (bad + ssm_a)[:20]))
    sys.exit(1)
if ssm_a:
    print(f"{n - len(ssm_a)} of {n} tensors match {tsv_path}; only {len(ssm_a)} ssm_a tensors differ, which comes from the "
          f"BF16 conversion, not from the recipe (BUILD.md, 'The BF16 conversion').")
    sys.exit(2)
print(f"OK: all {n} tensors match {tsv_path} (type and data SHA-256)")
