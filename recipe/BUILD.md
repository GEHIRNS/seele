# Rebuilding seele-1.0-3.3B

seele-1.0-3.3B is tattrongvu/Jeffan-4B-2010 with 8 of its 32 layers removed and the rest quantized, with no training.
- **Build:** one `llama-quantize` call (it prunes and quantizes together), then one metadata repair.
- **Release step:** writes the release metadata.
- **Hardware:** everything runs on the CPU, in a few minutes.

## Inputs

| Input | What it is | Check |
|---|---|---|
| `Jeffan-4B-2010-BF16.gguf` | [tattrongvu/Jeffan-4B-2010](https://huggingface.co/tattrongvu/Jeffan-4B-2010) at revision `937a008f86f4a20a903d628f22ffad3e3f431d2c`, converted to a BF16 GGUF with llama.cpp's `convert_hf_to_gguf.py` at tag `b11550` (commit `abee0c8`) | ours: 8,424,393,664 bytes, SHA-256 `a182f947e5c375aff113c33f0fd6261995704ebe77afbd1573548a9c6449690f`; see "The BF16 conversion" below |
| `imatrix-jeffan-nli-mmlu.gguf` | the importance matrix (this folder): 2,131 prompts in the parent's prompt format, 600 SNLI and MultiNLI training items and 1,531 MMLU validation questions | 992 tensors, 743 chunks of 512 tokens |
| `llama-quantize` | llama.cpp `b11550` (commit `abee0c8`). We built with it on Windows x64 and rebuilt on Linux x64; the quantized tensors came out identical | |
| `fix_pruned.py`, `set_release_metadata.py`, `verify_tensors.py` | this folder; they need `pip install gguf==0.19.0` | |

```sh
hf download tattrongvu/Jeffan-4B-2010 --revision 937a008f86f4a20a903d628f22ffad3e3f431d2c --local-dir Jeffan-4B-2010
git clone --branch b11550 --depth 1 https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py Jeffan-4B-2010 --outtype bf16 --outfile Jeffan-4B-2010-BF16.gguf
```

## Build

From the repository root:

```sh
llama-quantize --imatrix recipe/imatrix-jeffan-nli-mmlu.gguf \
  --token-embedding-type q4_k --tensor-type attn_qkv=q4_k \
  --tensor-type ffn_gate=iq4_xs --tensor-type ffn_up=iq4_xs --tensor-type ffn_down=iq4_xs \
  --prune-layers 20,21,23,24,25,26,28,29 \
  Jeffan-4B-2010-BF16.gguf seele-alpha-57.gguf Q4_K_M

python recipe/fix_pruned.py seele-alpha-57.gguf 20,21,23,24,25,26,28,29
python recipe/set_release_metadata.py seele-alpha-57.gguf seele-1.0-3.3B.gguf
```

What each step does:
- **`--prune-layers`** drops the 8 layers and renumbers the other 24.
  - **Which layers:** each of layers 12–30 was first removed alone, and the change in the model's answer distribution was measured as KL divergence from BF16 over 370 calibration requests. The six least damaging were removed together, then two more were added by greedy forward selection.
  - **What was removed:** seven Gated DeltaNet layers and one full-attention layer (23).
- **`Q4_K_M`** is the base mix. The overrides put the FFN matrices at IQ4_XS and the DeltaNet in-projection and token embedding at Q4_K. The token embedding is shared with the output layer.
- **`fix_pruned.py`:** llama-quantize b11550 leaves `qwen35.attention.recurrent_layers` at its unpruned length, so llama.cpp would refuse to load the file. This script removes the pruned entries and copies the tensors byte for byte.
- **`set_release_metadata.py`** writes:
  - the name, the size label (3.3B, the true total parameter count), the licence and the base models;
  - `qwen35.attention.causal = false`;
  - nothing for `quantize.imatrix.file` / `.dataset`, which it removes because they hold paths of the build machine.

| | Tensors | Parameters | Tensor data | Bits per weight |
|---|---|---|---|---|
| seele-1.0-3.3B | 317 | 3,307,799,744 | 1,812,626,176 bytes | 4.38 |

`seele-1.0-3.3B.tensor-types.tsv` lists every tensor with these columns:
- `ggml_type`
- `ne`: shape, in ggml order (innermost dimension first)
- `n_params` and `n_bytes`
- `group`
- `data_sha256`: SHA-256 of the tensor's data bytes in the released file

Composition:

| Group | Type | Tensors |
|---|---|---|
| FFN gate, up, down | IQ4_XS | 72 |
| DeltaNet in-projection (`attn_qkv`) | Q4_K | 17 |
| DeltaNet gate, out, alpha, beta | Q4_K | 68 |
| Full attention q, k, v, output | Q4_K | 24 |
| Full attention (part of v) | Q6_K | 4 |
| Token embedding (tied output head) | Q4_K | 1 |
| Norms, convolution, A, Δt | F32 | 131 |

## Verify

| File | Bytes | SHA-256 |
|---|---|---|
| seele-1.0-3.3B.gguf | 1,823,588,704 | `064302fa70f408c33859c121bc71a1eb2a23e506d58ad5ee830dcb9440fad3ac` |

```sh
python recipe/verify_tensors.py seele-1.0-3.3B.gguf recipe/seele-1.0-3.3B.tensor-types.tsv
```

It prints `OK: all 317 tensors match` when every tensor's type and data hash is the release's.

## The BF16 conversion

We rebuilt the release on a second machine (Linux) from a BF16 GGUF converted there from the same Hugging Face weights:
- **Metadata:** all 55 keys equal the release's.
- **Tensors:** 310 of 317 are byte-identical.
- **The other 7:** all `ssm_a`, an F32 tensor that `convert_hf_to_gguf.py` computes as −exp(A_log). The two BF16 conversions differ there and nowhere else. 12 of the 24 `ssm_a` tensors had 1–2 of their 32 values differ by one unit in the last place (largest relative difference 1.1 × 10⁻⁷). Every other tensor of the two conversions was identical.
- **Why:** the exponential rounds differently across numeric libraries.

`verify_tensors.py` reports that case separately, with exit code 2: `310 of 317 tensors match; only 7 ssm_a tensors differ`. A rebuild from a BF16 file with our SHA-256 should match the whole release byte for byte. We have not measured whether 1-ulp differences in `ssm_a` change any answer.
