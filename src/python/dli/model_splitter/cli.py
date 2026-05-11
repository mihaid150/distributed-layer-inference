from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from dli.model_splitter.splitter import split_model_into_stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or load a Hugging Face causal LM and split it into "
            "stage_1.pt, stage_2.pt, ... checkpoints for Distributed Layer Inference."
        )
    )

    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id to download/load.",
    )
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=None,
        help="Optional local model directory. If set, no HF download is attempted.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models/hf"),
        help="Project-local root used to store downloaded HF snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/partitions/tinyllama-1.1b-chat/4-stage"),
        help="Directory where stage_1.pt, stage_2.pt, ... will be written.",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=4,
        help="Number of stage checkpoints to create.",
    )
    parser.add_argument(
        "--layer-ranges",
        nargs="*",
        default=None,
        help=(
            "Optional explicit per-stage layer ranges. Examples: "
            "--layer-ranges 0-2 3-5 6-8 9-11 or "
            "--layer-ranges 0,1,2 3,4,5 6,7,8 9,10,11"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Torch dtype used when loading the model before splitting.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used during splitting. Use cpu on Raspberry Pi.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the HF snapshot into models-root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing stage checkpoint files.",
    )
    parser.add_argument(
        "--no-copy-tokenizer",
        action="store_true",
        help="Do not copy tokenizer/config into output-dir.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Optional path for split_manifest.json. Defaults to output-dir/split_manifest.json.",
    )
    parser.add_argument(
        "--stage-map-file",
        type=Path,
        default=Path("configs/stage_map.yaml"),
        help="Optional generated stage_map.yaml path. Use empty string to disable.",
    )
    parser.add_argument(
        "--physical-nodes",
        nargs="*",
        default=None,
        help="Optional node names written into generated stage_map.yaml, one per stage.",
    )

    args = parser.parse_args()

    if isinstance(args.stage_map_file, Path) and str(args.stage_map_file) == "":
        args.stage_map_file = None

    return args


def main() -> None:
    args = parse_args()

    plan = split_model_into_stages(
        model_id=args.model_id,
        local_model_dir=args.local_model_dir,
        models_root=args.models_root,
        output_dir=args.output_dir,
        num_stages=args.num_stages,
        layer_ranges=args.layer_ranges,
        dtype_name=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        force_download=args.force_download,
        force=args.force,
        copy_tokenizer=not args.no_copy_tokenizer,
        manifest_file=args.manifest_file,
        stage_map_file=args.stage_map_file,
        physical_nodes=args.physical_nodes,
    )

    print("Model split completed.")
    print(f"Model: {plan.model_id}")
    print(f"Source: {plan.model_source}")
    print(f"Output: {plan.output_dir}")
    print(f"Layers: {plan.num_layers}")
    print(f"Stages: {plan.num_stages}")
    for stage in plan.stages:
        print(
            f"  stage_{stage.stage_id}.pt -> layers={stage.layer_indices}, "
            f"embedding={stage.include_embedding}, "
            f"norm={stage.include_norm}, "
            f"lm_head={stage.include_lm_head}"
        )


if __name__ == "__main__":
    main()
