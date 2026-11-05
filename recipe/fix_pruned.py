"""Repair the layer metadata of a Qwen3.5 (qwen35) GGUF after `llama-quantize --prune-layers`.

llama-quantize (b11550) renumbers the remaining blocks and rewrites block_count, but leaves the per-layer array
qwen35.attention.recurrent_layers at its original length, so llama.cpp refuses to load the file ("wrong array length").
This rewrites the file with the array's entries for the pruned layers removed and block_count = the new array length
(the trailing tensor-less MTP entry is kept, as in the unpruned file). Tensors are copied byte for byte.

    pip install gguf==0.19.0
    python fix_pruned.py seele-1.0-3.3B.gguf 20,21,23,24,25,26,28,29     (the file is replaced in place)
"""
import importlib.util
import os
import pathlib
import sys

import gguf

_spec = importlib.util.spec_from_file_location("gnm", pathlib.Path(gguf.__file__).parent / "scripts" / "gguf_new_metadata.py")
gnm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gnm)


def main(path, layers):
    path = pathlib.Path(path)
    pruned = {int(x) for x in layers.split(",") if x}
    r = gguf.GGUFReader(str(path))
    arch = r.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    key = f"{arch}.attention.recurrent_layers"
    arr = r.get_field(key).contents()
    blocks = r.get_field(f"{arch}.block_count").contents()
    if len(arr) == blocks:
        print(f"{path.name}: already consistent ({blocks} blocks)")
        return 0
    new = [v for i, v in enumerate(arr) if i not in pruned]
    tmp = path.with_suffix(".fixing.gguf")
    w = gguf.GGUFWriter(str(tmp), arch=arch, endianess=r.endianess)
    gnm.copy_with_new_metadata(r, w, {key: gnm.MetadataDetails(gguf.GGUFValueType.ARRAY, new, sub_type=gguf.GGUFValueType.BOOL),
                                      f"{arch}.block_count": gnm.MetadataDetails(gguf.GGUFValueType.UINT32, len(new))}, [])
    del r, w
    os.replace(tmp, path)
    print(f"{path.name}: block_count {blocks} -> {len(new)}, {key} {len(arr)} -> {len(new)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
