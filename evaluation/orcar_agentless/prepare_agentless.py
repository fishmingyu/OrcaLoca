import argparse
import ast
import json
import os
from typing import Any, Dict, List

from Orcar.load_cache_dataset import load_filter_hf_dataset


def parse_line_range(line_range: str) -> List[int]:
    line_range = ast.literal_eval(line_range)
    try:
        assert len(line_range) == 2
        assert isinstance(line_range[0], int)
        assert isinstance(line_range[1], int)
        assert line_range[0] <= line_range[1]
    except AssertionError:
        print(f"Line Range Parse Error: {line_range}")
    return line_range


def parse_input(evaluation_path: str) -> List[Dict[str, Any]]:
    output_json_path = os.path.join(evaluation_path, "output.json")
    with open(output_json_path, "r") as f:
        output_json = json.load(f)
    # Disable the dependency output for now
    """
    dependency_output_json_path = os.path.join(
        evaluation_path, "dependency_output.json"
    )
    with open(dependency_output_json_path, "r") as f:
        dependency_output_json = json.load(f)
    """
    ret: List[Dict[str, Any]] = []
    for inst_id in output_json:
        bug_locations = output_json[inst_id]["bug_locations"]
        ret_inst = dict()
        ret_inst["instance_id"] = inst_id
        ret_inst["found_files"] = [x["file_path"] for x in bug_locations]
        ret_inst["found_edit_locs"] = dict()
        for x in bug_locations:
            if x["file_path"] not in ret_inst["found_edit_locs"]:
                ret_inst["found_edit_locs"][x["file_path"]] = []
            try:
                line_range = parse_line_range(x["line_range"])
            except AssertionError:
                print(f"Line Range Parse Error: {line_range} from instance {inst_id}")
                continue
            ret_inst["found_edit_locs"][x["file_path"]].append(
                f"line_range: {line_range[0]}-{line_range[1]}"
            )
        for k, v in ret_inst["found_edit_locs"].items():
            ret_inst["found_edit_locs"][k] = ["\n".join(v)]
        # Disable the dependency output for now
        """
        dependencies = dependency_output_json[inst_id]
        d_ret = dict()
        for d in dependencies:
            try:
                line_range = parse_line_range(d["line_range"])
            except AssertionError:
                print(f"Line Range Parse Error: {line_range} from instance {inst_id} ")
                continue
            d_ret[d["name"]] = {
                d["file_path"]: [f"line_range: {line_range[0]}-{line_range[1]}"]
            }

        ret_inst["dep_locs"] = d_ret
        """
        ret.append(ret_inst)
    return ret


def write_output(data: List[Dict[str, Any]], agentless_path: str) -> None:
    agentless_edit_loc_path = os.path.join(
        agentless_path, "results/swe-bench-lite/edit_location_individual"
    )
    os.makedirs(agentless_edit_loc_path, exist_ok=True)
    with open(
        os.path.join(agentless_edit_loc_path, "loc_orcar_outputs.jsonl"), "w"
    ) as f:
        for inst in data:
            f.write(json.dumps(inst) + "\n")
    inst_ids = {"target_inst_ids": [x["instance_id"] for x in data]}
    with open(os.path.join(agentless_path, "target_inst_ids.json"), "w") as f:
        json.dump(inst_ids, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluation_path", type=str, default="..")
    parser.add_argument(
        "-a", "--agentless_path", type=str, default="../../third_party/Agentless"
    )
    default_dataset = "princeton-nlp/SWE-bench_Lite"
    parser.add_argument(
        "-d",
        "--dataset",
        default=default_dataset,
        help=f"The target dataset (default: {default_dataset})",
    )
    parser.add_argument("-f", "--filter_instance", type=str, default="^(.*)$")
    parser.add_argument("-s", "--split", type=str, default="test")
    args = parser.parse_args()
    ds = load_filter_hf_dataset(args)
    insts = ds["instance_id"]
    data = parse_input(args.evaluation_path)
    print(f"Got number of instances: {len(data)}")
    data_filtered = [x for x in data if x["instance_id"] in insts]
    print(f"Filtered number of instances: {len(data_filtered)}")
    write_output(data_filtered, args.agentless_path)


if __name__ == "__main__":
    main()
