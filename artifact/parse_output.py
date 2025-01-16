import argparse
import json
import os
from typing import Dict, List

import gdown
import numpy as np
import pandas as pd
from pydantic import BaseModel


def download_golden_data(artifact_dir: str, dataset: str) -> pd.DataFrame:
    dataset_file = f"{dataset}_golden_stats.csv"
    dir = f"{artifact_dir}/assets"
    os.makedirs(dir, exist_ok=True)
    url_dict: Dict[str, str] = {
        "common_golden_stats.csv": "https://drive.google.com/file/d/1t0nl6LLq4WEVNHbasULLQ1onQQtxLL6A/view?usp=sharing",
        "lite_golden_stats.csv": "https://drive.google.com/file/d/1D9LWg78K66kHnex5RopBbTu_rwz1fhaS/view?usp=sharing",
        "verified_golden_stats.csv": "https://drive.google.com/file/d/1M6KZyLezU8ut0hrKUJBwi_TpvHV437NB/view?usp=sharing",
    }
    assert (
        dataset_file in url_dict
    ), f"Dataset file {dataset_file} not found in url_dict: {url_dict.keys()}"
    url = url_dict[dataset_file]
    file_dir = f"{dir}/{dataset_file}"
    if not os.path.isfile(file_dir):
        print(f"Downloading {dataset_file} from google drive url {url}")
        gdown.download(url, file_dir, quiet=False, fuzzy=True)
    return pd.read_csv(file_dir)


class DiffNode(BaseModel, frozen=True):
    node_name: str
    node_type: str
    lineno: int
    end_lineno: int

    def __repr__(self):
        return f"{self.node_type}:{self.node_name} {self.lineno}:{self.end_lineno}"


class SrcRange(BaseModel, frozen=True):
    lineno: int
    end_lineno: int
    is_pure_addition: bool
    is_global_addition: bool


class DiffLoc(BaseModel, frozen=True):
    file: str
    diff_nodes: List[DiffNode]
    lineno: int
    end_lineno: int

    def __repr__(self):
        return f"{self.file} {self.lineno}:{self.end_lineno}\n" + "\n".join(
            [
                f"    Node Level {i}: {repr(diff_node)}"
                for i, diff_node in enumerate(self.diff_nodes)
            ]
        )


class ParsedPatch(BaseModel):
    diff_locs: List[DiffLoc]

    def __repr__(self):
        return "\n".join(
            [
                f"Diff Loc {i} : {repr(diff_loc)}"
                for i, diff_loc in enumerate(self.diff_locs)
            ]
        )


def parse_output(ds_golden: pd.DataFrame, args) -> None:
    output_dir: str = args.output_dir
    artifact_dir: str = args.artifact_dir
    file_path_key: str = args.file_path_key
    file_match = 0
    keyword_match = 0
    notgen_cnt = 0
    # extractor_file_match = 0
    # extractor_notgen_cnt = 0
    output_dict = dict()
    issues = os.listdir(output_dir)
    golden_issues = set(ds_golden["instance_id"])
    issues = set(issues).intersection(golden_issues)
    issues = sorted(list(issues))
    file_snr_list = []
    keyword_snr_list = []
    for inst_id in sorted(issues):
        inst = ds_golden[ds_golden["instance_id"] == inst_id].iloc[0]
        output_dict[inst_id] = dict()

        parsed_patch = ParsedPatch.model_validate_json(inst["parsed_patch"])
        file_set = set()
        keyword_set = set()
        for diff_loc in parsed_patch.diff_locs:
            file_path = diff_loc.file
            file_set.add(file_path)
            diff_nodes = diff_loc.diff_nodes
            keyword_name = file_path + ":"
            if len(diff_nodes) == 0:
                continue
            keyword_name += diff_nodes[0].node_name
            if (
                len(diff_nodes) > 1
                and diff_nodes[0].node_type == "ClassDef"
                and diff_nodes[1].node_type == "FunctionDef"
            ):
                keyword_name += "." + diff_nodes[1].node_name
            elif len(diff_nodes) > 1 and diff_nodes[0].node_type != "FunctionDef":
                print("Weird diff_loc:", inst_id, diff_nodes)
                continue
            keyword_set.add(keyword_name)
        if not file_set:
            print("No file found", inst_id)
            print(parsed_patch)
        # if not keyword_set:
        #    print('No keyword found', inst_id)
        #    print(parsed_patch)

        # is_searcher_match = False
        json_dir = f"{output_dir}/{inst_id}/searcher_{inst_id}.json"
        print(f"Checking {inst_id}")
        model_file_set = set()
        model_keyword_set = set()
        # print(inst_id)
        if not os.path.isfile(json_dir):
            notgen_cnt += 1
            output_dict[inst_id]["status"] = "Json not gen"
            # print('    Json not gen')
        else:
            with open(json_dir, "r") as handle:
                model_searcher_output = json.load(handle)
            if "bug_locations" not in model_searcher_output:
                notgen_cnt += 1
                output_dict[inst_id]["status"] = "Json invalid"
            else:
                for loc in model_searcher_output["bug_locations"]:
                    file_path = loc[file_path_key]
                    if file_path and file_path[0] == "/":
                        file_path = file_path[1:]
                    model_file_set.add(file_path)
                    keyword = loc[file_path_key] + ":"
                    if not (bool(loc["class_name"]) or bool(loc["method_name"])):
                        continue
                    elif not loc["class_name"]:
                        keyword += loc["method_name"]
                    elif not loc["method_name"]:
                        keyword += loc["class_name"]
                    else:
                        model_keyword_set.add(keyword + loc["class_name"])
                        keyword += loc["class_name"] + "." + loc["method_name"]
                    model_keyword_set.add(keyword)
                output_dict[inst_id]["file"] = dict()
                if file_set.issubset(model_file_set):
                    file_match += 1
                    output_dict[inst_id]["file"]["file_status"] = "Matched"
                else:
                    output_dict[inst_id]["file"]["file_status"] = "Not Matched"
                file_snr_list.append(
                    len(file_set.intersection(model_file_set)) / len(model_file_set)
                    if len(model_file_set)
                    else 1
                )

                output_dict[inst_id]["file"]["golden"] = list(file_set)
                output_dict[inst_id]["file"]["model"] = list(model_file_set)

                output_dict[inst_id]["keyword"] = dict()
                if keyword_set.issubset(model_keyword_set):
                    keyword_match += 1

                    output_dict[inst_id]["keyword"]["keyword_status"] = "Matched"
                else:
                    output_dict[inst_id]["keyword"]["keyword_status"] = "Not Matched"
                output_dict[inst_id]["keyword"]["golden"] = list(keyword_set)
                output_dict[inst_id]["keyword"]["model"] = list(model_keyword_set)
                keyword_snr_list.append(
                    len(keyword_set.intersection(model_keyword_set))
                    / len(model_keyword_set)
                    if len(model_keyword_set)
                    else 1
                )

    total_cnt = len(issues)
    print(f"File match: {file_match}/{total_cnt}, {file_match / total_cnt * 100:.2f}%")
    print(
        f"Mean File SNR: {np.mean(file_snr_list):.2f}, Std File SNR: {np.std(file_snr_list):.2f}"
    )
    print(
        f"Keyword Match: {keyword_match}/{total_cnt}, {keyword_match / total_cnt * 100:.2f}%"
    )
    print(
        f"Mean Keyword SNR: {np.mean(keyword_snr_list):.2f}, Std Keyword SNR: {np.std(keyword_snr_list):.2f}"
    )
    print(
        f"Json not gen: {notgen_cnt}/{total_cnt}, {notgen_cnt / total_cnt * 100:.2f}%"
    )
    output_path = f"{artifact_dir}/assets/orcar_parsed_output.json"
    with open(output_path, "w") as handle:
        json.dump(output_dict, handle, indent=4)
    print(f"Parsed output dumped to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--artifact_dir",
        default="./artifact",
        help=f"The directory of the artifact folder",
    )
    parser.add_argument(
        "-l",
        "--output_dir",
        default="./output",
        help=f"The directory of the output dir(agent's output)",
    )
    parser.add_argument(
        "-f",
        "--file_path_key",
        default="file_path",
        help=f"The directory of the output dir(agent's output)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="lite",
        help=f"The dataset to use",
    )
    args = parser.parse_args()
    ds_golden = download_golden_data(
        artifact_dir=args.artifact_dir, dataset=args.dataset
    )
    parse_output(ds_golden, args)


if __name__ == "__main__":
    main()
