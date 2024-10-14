import argparse
import json
import os
from typing import Dict, List

import gdown
import pandas as pd
from pydantic import BaseModel


def download_golden_data(artifact_dir: str) -> pd.DataFrame:
    dir = f"{artifact_dir}/assets"
    os.makedirs(dir, exist_ok=True)
    url_dict: Dict[str, str] = {
        "common_golden_stats.csv": "https://drive.google.com/file/d/1t0nl6LLq4WEVNHbasULLQ1onQQtxLL6A/view?usp=sharing",
        "lite_golden_stats.csv": "https://drive.google.com/file/d/1D9LWg78K66kHnex5RopBbTu_rwz1fhaS/view?usp=sharing",
        "verified_golden_stats.csv": "https://drive.google.com/file/d/1M6KZyLezU8ut0hrKUJBwi_TpvHV437NB/view?usp=sharing",
    }
    csv_list: List[pd.DataFrame] = []
    for filename, url in url_dict.items():
        file_dir = f"{dir}/{filename}"
        if not os.path.isfile(file_dir):
            print(f"Downloading {filename} from google drive url {url}")
            gdown.download(url, file_dir, quiet=False, fuzzy=True)
        csv_list.append(pd.read_csv(file_dir))

    if not csv_list:
        return pd.DataFrame()
    ret = csv_list[0]

    if len(csv_list) == 1:
        return csv_list[0]
    for csv in csv_list[1:]:
        ret = ret.merge(csv, how="outer", validate="1:1").reset_index(drop=True)
    return ret


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


def parse_log(ds_golden: pd.DataFrame, log_dir: str, artifact_dir: str) -> None:
    file_match = 0
    keyword_match = 0
    notgen_cnt = 0
    extractor_file_match = 0
    extractor_notgen_cnt = 0
    log_dict = dict()
    issues = os.listdir(log_dir)
    for inst_id in sorted(issues):
        inst = ds_golden[ds_golden["instance_id"] == inst_id].iloc[0]
        log_dict[inst_id] = dict()

        parsed_patch = ParsedPatch.model_validate_json(inst["parsed_patch"])
        file_set = set()
        keyword_set = set()
        for diff_loc in parsed_patch.diff_locs:
            file_name = diff_loc.file
            file_set.add(file_name)
            diff_nodes = diff_loc.diff_nodes
            keyword_name = file_name + ":"
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
        json_dir = f"{log_dir}/{inst_id}/searcher_{inst_id}.json"
        model_file_set = set()
        model_keyword_set = set()
        # print(inst_id)
        if not os.path.isfile(json_dir):
            notgen_cnt += 1
            log_dict[inst_id]["status"] = "Json not gen"
            # print('    Json not gen')
        else:
            with open(json_dir, "r") as handle:
                model_searcher_output = json.load(handle)
            if "bug_locations" not in model_searcher_output:
                notgen_cnt += 1
                log_dict[inst_id]["status"] = "Json invalid"
            else:
                for loc in model_searcher_output["bug_locations"]:
                    file_name = loc["file"]
                    if file_name and file_name[0] == "/":
                        file_name = file_name[1:]
                    model_file_set.add(file_name)
                    keyword = loc["file"] + ":"
                    if not (bool(loc["class"]) or bool(loc["method"])):
                        continue
                    elif not loc["class"]:
                        keyword += loc["method"]
                    elif not loc["method"]:
                        keyword += loc["class"]
                    else:
                        keyword += loc["class"] + "." + loc["method"]
                    model_keyword_set.add(keyword)
                if file_set.issubset(model_file_set):
                    file_match += 1
                    # is_searcher_match = True
                    # print('    File Matched!')
                    log_dict[inst_id]["file"] = "Matched"
                else:
                    # print('    File mismatch:')
                    # print('    Golden: \n    ',file_set)
                    # print('    Model: \n    ',model_file_set)
                    log_dict[inst_id]["file"] = dict()
                    log_dict[inst_id]["file"]["golden"] = list(file_set)
                    log_dict[inst_id]["file"]["model"] = list(model_file_set)

                if keyword_set.issubset(model_keyword_set):
                    keyword_match += 1
                    # print('    Keyword Matched!')
                    log_dict[inst_id]["keyword"] = "Matched"
                else:
                    # print('    Keyword mismatch:')
                    # print('    Golden: \n    ',keyword_set)
                    # print('    Model: \n    ',model_keyword_set)
                    log_dict[inst_id]["keyword"] = dict()
                    log_dict[inst_id]["keyword"]["golden"] = list(keyword_set)
                    log_dict[inst_id]["keyword"]["model"] = list(model_keyword_set)

        # is_extractor_match = False
        json_dir = f"{log_dir}/{inst_id}/extractor_{inst_id}.json"
        extractor_file_set = set()
        if not os.path.isfile(json_dir):
            extractor_notgen_cnt += 1
            # print('Not gen:', inst_id)
        else:
            with open(json_dir, "r") as handle:
                model_extractor_output = json.load(handle)
            for code in model_extractor_output["suspicious_code"]:
                file_name = code["file_path"]
                if file_name and file_name[0] == "/":
                    file_name = file_name[1:]
                extractor_file_set.add(file_name)
            if file_set.issubset(extractor_file_set):
                extractor_file_match += 1
                # is_extractor_match = True
        # if is_extractor_match and not is_searcher_match:
        # print(f'Extractor match but searcher missed: {inst_id}')

    total_cnt = len(issues)
    print(f"File match: {file_match}/{total_cnt}, {file_match / total_cnt * 100:.2f}%")
    print(
        f"Keyword Match: {keyword_match}/{total_cnt}, {keyword_match / total_cnt * 100:.2f}%"
    )
    print(
        f"Json not gen: {notgen_cnt}/{total_cnt}, {notgen_cnt / total_cnt * 100:.2f}%"
    )
    # print(f"Extractor File match: {extractor_file_match / total_cnt:.2f}")
    # print(f"Extractor Json not gen: {extractor_notgen_cnt / total_cnt:.2f}")
    output_path = f"{artifact_dir}/assets/orcar_parsed_log.json"
    with open(output_path, "w") as handle:
        json.dump(log_dict, handle, indent=4)
    print(f"Parsed log dumped to {output_path}")


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
        "--log_dir",
        default="./log",
        help=f"The directory of the log dir",
    )
    args = parser.parse_args()
    log_dir: str = args.log_dir
    artifact_dir: str = args.artifact_dir
    ds_golden = download_golden_data(artifact_dir=artifact_dir)
    parse_log(ds_golden, log_dir=log_dir, artifact_dir=artifact_dir)


if __name__ == "__main__":
    main()
