import glob
import json
import os
from argparse import ArgumentParser, Namespace

import swebench.harness.utils
from swebench.harness.run_evaluation import main as run_evaluation_main

from Orcar.log_utils import get_logger

logger = get_logger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Lite",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    # parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        type=swebench.harness.utils.str2bool,
        default=False,
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean",
        type=swebench.harness.utils.str2bool,
        default=False,
        help="Clean images above cache level",
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )

    parser.add_argument(
        "--orcar_root_path",
        type=str,
        help="Path to orcar root. Will collect output from root/output and write result to root/artifact/assets",
        required=True,
    )
    return parser.parse_args()


def parse_output(output_path: str, prediction_path: str) -> None:
    # Get all patch files
    patch_files = glob.glob(
        os.path.join(output_path, "**/editor_*.patch"), recursive=True
    )

    # Process each file and write to jsonl
    with open(prediction_path, "w") as f:
        for patch_file in patch_files:
            # Extract instance_id from path
            instance_id = (
                os.path.basename(patch_file)
                .replace("editor_", "")
                .replace(".patch", "")
            )

            # Read patch content
            with open(patch_file, "r") as pf:
                patch_content = pf.read()

            # Create output entry
            entry = {
                "instance_id": instance_id,
                "model_name_or_path": "orcar",
                "model_patch": patch_content,
            }

            # Write to jsonl file
            f.write(json.dumps(entry) + "\n")


def main():
    args = get_args()
    orcar_root_path = args.orcar_root_path
    delattr(args, "orcar_root_path")
    assets_path = os.path.abspath(os.path.join(orcar_root_path, "artifact", "assets"))
    prediction_path = os.path.abspath(os.path.join(assets_path, "all_preds.jsonl"))
    args.predictions_path = prediction_path
    output_path = os.path.join(orcar_root_path, "output")
    parse_output(output_path, prediction_path)
    original_cwd = os.getcwd()
    try:
        os.chdir(assets_path)
        run_evaluation_main(**vars(args))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
