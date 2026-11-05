"""Write the release metadata of seele-1.0-3.3B into a rebuilt GGUF (a new file; tensors are copied byte for byte).

    pip install gguf==0.19.0
    python set_release_metadata.py seele-alpha-57.gguf seele-1.0-3.3B.gguf

Sets the name, size label (the true total parameter count), version, licence, description and base models; sets
qwen35.attention.causal = false (the model reads the whole prompt bidirectionally, as its parent); removes
quantize.imatrix.file and quantize.imatrix.dataset (paths of the build machine). Refuses to write a file whose tensors
differ from the input or whose string metadata still holds a local path.
"""
import hashlib
import importlib.util
import pathlib
import re
import sys

import gguf

_spec = importlib.util.spec_from_file_location("gnm", pathlib.Path(gguf.__file__).parent / "scripts" / "gguf_new_metadata.py")
gnm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gnm)
S = gguf.GGUFValueType.STRING
LOCAL = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])|[\\/]Users[\\/]|[\\/]home[\\/]")
TITLE = "seele-1.0-3.3B"
RECIPE = ("Layers 20, 21, 23, 24, 25, 26, 28 and 29 of 32 removed (llama-quantize --prune-layers), chosen one at a time by greedy "
          "forward selection on output divergence (KL) from BF16; the remaining weights quantized with llama.cpp b11550 (Q4_K_M base; "
          "FFN gate/up/down IQ4_XS; attention QKV, DeltaNet projections and token embedding Q4_K; 4.38 bits per weight) with an "
          "importance matrix from 2,131 SNLI/MultiNLI train and MMLU validation prompts. No training.")


def digest(reader):
    h = hashlib.sha256()
    for t in reader.tensors:
        h.update(t.name.encode())
        h.update(memoryview(t.data).cast("B"))
    return h.hexdigest()


def main(src, dst):
    reader = gguf.GGUFReader(src)
    arch = reader.get_field("general.architecture").contents()
    new = {
        "general.name": gnm.MetadataDetails(S, TITLE),
        "general.basename": gnm.MetadataDetails(S, "SEELE"),
        "general.version": gnm.MetadataDetails(S, "1.0"),
        "general.license": gnm.MetadataDetails(S, "apache-2.0"),
        "general.description": gnm.MetadataDetails(S, f"{TITLE}: a fast low-memory derivative of tattrongvu/Jeffan-4B-2010 @937a008 "
                                                       f"(Apache-2.0). {RECIPE}"),
        "general.base_model.count": gnm.MetadataDetails(gguf.GGUFValueType.UINT32, 2),
        "general.base_model.0.name": gnm.MetadataDetails(S, "Jeffan 4B 2010"),
        "general.base_model.0.organization": gnm.MetadataDetails(S, "tattrongvu"),
        "general.base_model.0.version": gnm.MetadataDetails(S, "937a008"),
        "general.base_model.0.repo_url": gnm.MetadataDetails(S, "https://huggingface.co/tattrongvu/Jeffan-4B-2010"),
        "general.base_model.1.name": gnm.MetadataDetails(S, "Qwen3.5 4B"),
        "general.base_model.1.organization": gnm.MetadataDetails(S, "Qwen"),
        "general.base_model.1.repo_url": gnm.MetadataDetails(S, "https://huggingface.co/Qwen/Qwen3.5-4B"),
        "general.size_label": gnm.MetadataDetails(S, "3.3B"),
        f"{arch}.attention.causal": gnm.MetadataDetails(gguf.GGUFValueType.BOOL, False),
    }
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    align = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if align:
        writer.data_alignment = align.contents()
    gnm.copy_with_new_metadata(reader, writer, new, ["quantize.imatrix.file", "quantize.imatrix.dataset"])
    out = gguf.GGUFReader(dst)
    same = digest(reader) == digest(out)
    leaks = [k for k, f in out.fields.items() if f.types and f.types[0] == S and LOCAL.search(str(f.contents()))]
    del out, reader, writer
    if not same or leaks:
        pathlib.Path(dst).unlink()
        raise SystemExit(f"check failed: tensors identical={same}, local paths in {leaks}")
    print(f"{dst}: release metadata written, tensors identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
