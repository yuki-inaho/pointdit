# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert the gated DINOv3 Hugging Face safetensors to the upstream ``.pth`` layout.

The DINOv3 encoder weights are gated and the FAIR download URLs are not public;
the official Hugging Face mirrors (``facebook/dinov3-vit{l,b,h}16-...``) are
gated as well but can be downloaded after access is granted.  PointDiT expects
the upstream raw state-dict layout (``cls_token``, ``storage_tokens``,
``patch_embed.proj.*``, fused ``blocks.N.attn.qkv.*``, ...), so this script
re-keys the HF tensors and adds the deterministic buffers (``rope_embed.periods``
and the qkv ``bias_mask``) taken from a freshly built upstream backbone.

Example:
    python scripts/convert_dinov3_hf_weights.py \
        --input pretrained/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.hf.safetensors \
        --output pretrained/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --backbone dinov3_vitl16
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "third_party" / "dinov3"))

from dinov3.hub import backbones  # noqa: E402

_SIMPLE = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.register_tokens": "storage_tokens",
    "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
}
_LAYER_SUFFIX = {
    "attention.o_proj.weight": "attn.proj.weight",
    "attention.o_proj.bias": "attn.proj.bias",
    "layer_scale1.lambda1": "ls1.gamma",
    "layer_scale2.lambda1": "ls2.gamma",
    "mlp.up_proj.weight": "mlp.fc1.weight",
    "mlp.up_proj.bias": "mlp.fc1.bias",
    "mlp.down_proj.weight": "mlp.fc2.weight",
    "mlp.down_proj.bias": "mlp.fc2.bias",
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
}


def convert(hf: dict, template: dict) -> dict:
    converted: dict = {}
    qkv: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in hf.items():
        if key in _SIMPLE:
            converted[_SIMPLE[key]] = value
            continue
        if not key.startswith("layer."):
            raise KeyError(f"unmapped HF key: {key}")
        _, index, suffix = key.split(".", 2)
        if suffix.startswith("attention.") and "_proj" in suffix and suffix in {
            "attention.q_proj.weight",
            "attention.q_proj.bias",
            "attention.k_proj.weight",
            "attention.k_proj.bias",
            "attention.v_proj.weight",
            "attention.v_proj.bias",
        }:
            qkv.setdefault(index, {})[suffix] = value
            continue
        if suffix not in _LAYER_SUFFIX:
            raise KeyError(f"unmapped HF key: {key}")
        converted[f"blocks.{index}.{_LAYER_SUFFIX[suffix]}"] = value

    for index, parts in qkv.items():
        for tensor in ("weight", "bias"):
            pieces = []
            for name in ("q", "k", "v"):
                piece = parts.get(f"attention.{name}_proj.{tensor}")
                if piece is None:
                    # The HF mirror drops the key bias: the upstream forward zeroes
                    # it anyway via ``qkv.bias_mask`` (middle third masked out).
                    if name != "k" or tensor != "bias":
                        raise KeyError(f"layer {index}: missing {name}_proj.{tensor}")
                    reference = parts.get("attention.q_proj.bias")
                    if reference is None:
                        raise KeyError(f"layer {index}: no reference tensor for the key bias")
                    piece = torch.zeros_like(reference)
                pieces.append(piece)
            converted[f"blocks.{index}.attn.qkv.{tensor}"] = torch.cat(pieces, dim=0)

    for key, value in template.items():
        if key in converted:
            continue
        if key.endswith("qkv.bias_mask") or key.endswith("rope_embed.periods"):
            converted[key] = value
            continue
        raise KeyError(f"template key {key} is not covered by the HF mapping")
    missing = sorted(set(template) - set(converted))
    if missing:
        raise KeyError(f"converted dict is missing template keys: {missing}")
    for key, value in list(converted.items()):
        target = template[key]
        if tuple(value.shape) != tuple(target.shape):
            if value.numel() != target.numel():
                raise ValueError(
                    f"{key}: cannot reshape {tuple(value.shape)} into {tuple(target.shape)}"
                )
            converted[key] = value.reshape(target.shape)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov3_vitl16")
    args = parser.parse_args()

    hf = load_file(str(args.input))
    backbone = getattr(backbones, args.backbone)(pretrained=False)
    template = backbone.state_dict()
    converted = convert(hf, template)
    if sorted(converted) != sorted(template):
        raise SystemExit("converted keys do not match the upstream layout")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"converted {len(converted)} tensors -> {args.output}")

    reloaded = getattr(backbones, args.backbone)(pretrained=True, weights=str(args.output))
    del reloaded
    print("strict load check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
