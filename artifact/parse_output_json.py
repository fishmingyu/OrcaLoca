import argparse
import json

import numpy as np
import pandas as pd
from parse_output import ParsedPatch, download_golden_data


def parse_output_json(ds_golden: pd.DataFrame, args) -> None:
    output_json = json.load(open(args.output_json))
    artifact_dir: str = args.artifact_dir
    file_path_key: str = args.file_path_key
    file_match = 0
    func_match = 0
    notgen_cnt = 0
    # extractor_file_match = 0
    # extractor_notgen_cnt = 0
    output_dict = dict()
    issues = set(output_json.keys())
    file_prec_list = []
    func_prec_list = []
    output_json = json.load(open(args.output_json))
    for inst_id in sorted(issues):
        inst = ds_golden[ds_golden["instance_id"] == inst_id].iloc[0]
        output_dict[inst_id] = dict()

        parsed_patch = ParsedPatch.model_validate_json(inst["parsed_patch"])
        file_set = set()
        func_set = set()
        for diff_loc in parsed_patch.diff_locs:
            file_path = diff_loc.file
            file_set.add(file_path)
            diff_nodes = diff_loc.diff_nodes
            func_name = file_path + ":"
            if len(diff_nodes) == 0:
                continue
            func_name += diff_nodes[0].node_name
            if (
                len(diff_nodes) > 1
                and diff_nodes[0].node_type == "ClassDef"
                and diff_nodes[1].node_type == "FunctionDef"
            ):
                func_name += "." + diff_nodes[1].node_name
            elif len(diff_nodes) > 1 and diff_nodes[0].node_type != "FunctionDef":
                print("Weird diff_loc:", inst_id, diff_nodes)
                continue
            func_set.add(func_name)
        if not file_set:
            print("No file found", inst_id)
            print(parsed_patch)

        model_file_set = set()
        model_func_set = set()
        # print(inst_id)

        instance_info = output_json[inst_id]
        if "bug_locations" not in instance_info:
            notgen_cnt += 1
            output_dict[inst_id]["status"] = "Json Not Gen"
            continue
        else:
            model_searcher_output = instance_info
            for loc in model_searcher_output["bug_locations"]:
                file_path = loc[file_path_key]
                if file_path and file_path[0] == "/":
                    file_path = file_path[1:]
                model_file_set.add(file_path)
                func = loc[file_path_key] + ":"
                if not (bool(loc["class_name"]) or bool(loc["method_name"])):
                    continue
                elif not loc["class_name"]:
                    func += loc["method_name"]
                elif not loc["method_name"]:
                    func += loc["class_name"]
                else:
                    model_func_set.add(func + loc["class_name"])
                    func += loc["class_name"] + "." + loc["method_name"]
                model_func_set.add(func)
            output_dict[inst_id]["file"] = dict()
            if file_set.issubset(model_file_set):
                file_match += 1
                output_dict[inst_id]["file"]["file_status"] = "Matched"
            else:
                output_dict[inst_id]["file"]["file_status"] = "Not Matched"
            file_prec_list.append(
                len(file_set.intersection(model_file_set)) / len(model_file_set)
                if len(model_file_set)
                else 1
            )

            output_dict[inst_id]["file"]["golden"] = list(file_set)
            output_dict[inst_id]["file"]["model"] = list(model_file_set)

            output_dict[inst_id]["func"] = dict()
            if func_set.issubset(model_func_set):
                func_match += 1

                output_dict[inst_id]["func"]["func_status"] = "Matched"
            else:
                output_dict[inst_id]["func"]["func_status"] = "Not Matched"
            output_dict[inst_id]["func"]["golden"] = list(func_set)
            output_dict[inst_id]["func"]["model"] = list(model_func_set)
            func_prec_list.append(
                len(func_set.intersection(model_func_set)) / len(model_func_set)
                if len(model_func_set)
                else 1
            )

    total_cnt = len(issues)
    print(f"File match: {file_match}/{total_cnt}, {file_match / total_cnt * 100:.2f}%")
    print(
        f"Mean File Precision: {np.mean(file_prec_list) * 100:.2f}%, Std File Precision: {np.std(file_prec_list) * 100:.2f}%"
    )
    print(
        f"Function Match: {func_match}/{total_cnt}, {func_match / total_cnt * 100:.2f}%"
    )
    print(
        f"Mean Function Precision: {np.mean(func_prec_list) * 100:.2f}%, Std Function Precision: {np.std(func_prec_list) * 100:.2f}%"
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
        "--output_json",
        default="./evaluation/output.json",
        help=f"The file path of the output json",
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
    parse_output_json(ds_golden, args)


if __name__ == "__main__":
    main()
