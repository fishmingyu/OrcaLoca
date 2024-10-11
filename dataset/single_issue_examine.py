import argparse
import json
import os
import subprocess
from typing import Any, Dict

import unidiff
import yaml

from Orcar.load_cache_dataset import load_filter_hf_dataset_explicit
from Orcar.log_utils import get_logger

logger = get_logger(__name__)

"""
Example Usage:
    python dataset/single_issue_examine.py --instance_id astropy__astropy-12907
    python dataset/single_issue_examine.py --instance_id 'astropy__astropy-12907' --experiment_dir '~/sandbox/RAGCompiler/swe_bench_stats/experiments'

"""


def get_instance(dataset: str, instance_id: str) -> Dict[str, Any]:
    ds = load_filter_hf_dataset_explicit(
        dataset=dataset, filter_instance=instance_id, split="test"
    )
    assert len(ds) > 0, f"Error: Cannot find {instance_id} in {dataset}"
    assert (
        len(ds) < 2
    ), f"Error: Found multiple instance with ID {instance_id} in {dataset}"
    return ds[0]


def examine_experiment_dir(
    inst: Dict[str, Any], experiment_dir: str, dataset: str, verbose: int
) -> None:
    dataset_dir_dict = {
        "princeton-nlp/SWE-bench": "test",
        "princeton-nlp/SWE-bench_Lite": "lite",
        "princeton-nlp/SWE-bench_Verified": "verified",
    }
    assert dataset in dataset_dir_dict, "Error: Unknown dataset {dataset}"
    dataset_dir = f"{experiment_dir}/evaluation/{dataset_dir_dict[dataset]}"
    assert os.path.isdir(dataset_dir), "Error: Invalid dataset dir {dataset_dir}"
    subprocess.run("git reset --hard main", shell=True, check=True, cwd=experiment_dir)
    subprocess.run("git pull", shell=True, check=True, cwd=experiment_dir)
    models = sorted(os.listdir(dataset_dir))
    solved_cnt = 0
    for model in models:
        date = model.split("_")[0]
        model_path = f"{dataset_dir}/{model}"
        results_path = f"{model_path}/results/results.json"
        if not os.path.exists(results_path):
            logger.error(f"Cannot find {results_path}")
            continue
        with open(results_path, "r") as file:
            results_json = json.load(file)

        metadata_path = f"{model_path}/metadata.yaml"
        assert os.path.exists(metadata_path)
        with open(metadata_path) as stream:
            try:
                metadata = yaml.safe_load(stream)
                model_name = metadata["name"]
            except yaml.YAMLError as exc:
                print(exc)

        if inst["instance_id"] not in results_json["resolved"]:
            continue

        logger.info(f"{model_name} \n    Solved ✓ on {date}")
        solved_cnt += 1

        preds_path = f"{model_path}/all_preds.jsonl"
        assert os.path.exists(preds_path)
        model_inst = dict()
        with open(preds_path, "r") as file:
            for line in file:
                issue_json = json.loads(line)
                if issue_json["instance_id"] == inst["instance_id"]:
                    model_inst = issue_json
                    break
        if not model_inst:
            logger.warning(
                f"Cannot find {inst['instance_id']} in {model_name}, skipping"
            )

        if verbose > 0:
            logger.info("Model Patch:")
            logger.info(model_inst["model_patch"])
            continue

        try:
            model_patch = unidiff.PatchSet(model_inst["model_patch"])
        except Exception as e:
            logger.warning(
                (
                    f"Returning empty patch,",
                    f" as patch for {model_inst['instance_id']}",
                    f" is corrupted with msg '{repr(e)}'",
                )
            )
            continue
        abs_patch = []
        abs_patch.append(" " * 4 + "Model Patch:")
        for file in model_patch:
            assert isinstance(file, unidiff.PatchedFile)
            abs_patch.append(" " * 4 + str(file.patch_info).split("\n")[0].strip())
            for hunk in file:
                abs_patch.append(" " * 4 + str(hunk).split("\n")[0])
        logger.info("\n".join(abs_patch))
    logger.info(
        f"Issue {inst['instance_id']} is solved by {solved_cnt}/{len(models)} models"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--instance_id",
        required=True,
        help=f"The ID of target instance",
    )
    default_dataset = "princeton-nlp/SWE-bench_Lite"
    parser.add_argument(
        "-d",
        "--dataset",
        default=default_dataset,
        help=f"The target dataset (default: {default_dataset})",
    )
    parser.add_argument(
        "-e",
        "--experiment_dir",
        default=None,
        help=f"The directory of SWE-bench Experiments (default: None)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        help=f"The directory of SWE-bench Experiments (default: None)",
    )
    args = parser.parse_args()
    inst = get_instance(dataset=args.dataset, instance_id=args.instance_id)
    logger.info(f"Found instance        {args.instance_id}:")
    logger.info(f"Repo:                 {inst['repo']}")
    logger.info(f"Version:              {inst['version']}")
    logger.info(f"Base Commit:          {inst['base_commit']}")
    if args.verbose > 0:
        logger.info(f"Problem statement:")
        logger.info(f"{inst['problem_statement']}")
        logger.info(f"Golden Patch:")
        logger.info(f"{inst['patch']}")
    else:
        logger.info(f"Golden Patch:")
        patch = unidiff.PatchSet(inst["patch"])
        abs_patch = []
        for file in patch:
            abs_patch.append(" " * 4 + str(file.patch_info).strip())
            for hunk in file:
                abs_patch.append(" " * 4 + str(hunk).split("\n")[0])
        logger.info("\n".join(abs_patch))
    if args.experiment_dir:
        assert os.path.isdir(args.experiment_dir)
        examine_experiment_dir(
            inst=inst,
            experiment_dir=args.experiment_dir,
            dataset=args.dataset,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
