from datasets import load_dataset
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Set
import subprocess
import ast
import unidiff
from unidiff import PatchSet
import os
from tqdm import tqdm


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


def get_diff_nodes_from_file(
    filename: str, src_range: SrcRange, dbg: bool = False
) -> List[DiffNode]:
    lineno = src_range.lineno
    end_lineno = src_range.end_lineno
    with open(filename, "r") as file:
        source_code = file.read()

    node = ast.parse(source_code)
    ret = []
    while True:
        for child in ast.iter_child_nodes(node):
            if child.__class__.__name__ not in ["FunctionDef", "ClassDef"]:
                continue
            if (child.lineno <= lineno) and (child.end_lineno >= end_lineno):
                ret.append(
                    DiffNode(
                        node_name=child.name,
                        node_type=child.__class__.__name__,
                        lineno=child.lineno,
                        end_lineno=child.end_lineno,
                    )
                )
                node = child
                break
        else:
            break
    if src_range.is_pure_addition and ret:
        first_append_idx = -1
        for i, node in enumerate(ret):
            if dbg:
                print(
                    i,
                    lineno,
                    node,
                    lineno == node.end_lineno,
                    node.node_name == "FunctionDef",
                )
            if lineno == node.end_lineno and node.node_type == "FunctionDef":
                first_append_idx = i
                break
        if (first_append_idx == 0) and (not src_range.is_global_addition):
            return ret[0:1]
        if first_append_idx != -1:
            return ret[0:first_append_idx]
    return ret


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


def get_src_range_from_hunk(
    hunk: unidiff.patch.Hunk, dbg: bool = False
) -> List[SrcRange]:
    idx = hunk.source_start
    last_unchanged_idx = idx
    last_unchanged_nonblank_idx = idx
    modified_lines = []
    src_ranges = []
    found_removal = False
    is_global_addition = False
    got_first_addition = False
    for line in hunk:
        if line.is_added:
            if found_removal:
                modified_lines.append(last_unchanged_idx + 1)
            else:
                modified_lines.append(last_unchanged_nonblank_idx)
                if line.value.strip:
                    if (line.value.lstrip() == line.value) and (not got_first_addition):
                        is_global_addition = True
                    got_first_addition = True
            continue
        if line.is_removed:
            modified_lines.append(idx)
            found_removal = True
        else:
            if modified_lines:
                if dbg:
                    print(modified_lines)
                modified_lines = sorted(modified_lines)
                assert not (found_removal and is_global_addition)
                src_ranges.append(
                    SrcRange(
                        lineno=modified_lines[0],
                        end_lineno=modified_lines[-1],
                        is_pure_addition=not found_removal,
                        is_global_addition=is_global_addition,
                    )
                )
                modified_lines = []
            if line.value.strip():
                last_unchanged_nonblank_idx = idx
            last_unchanged_idx = idx
            found_removal = False
            is_global_addition = False
            got_first_addition = False
            level = 0

        idx += 1
    return src_ranges


def parse_patch(
    patch: str,
    repo: str,
    base_commit: str,
    instance_id: str,
    base: str,
    dbg: bool = False,
) -> str:
    diff_locs: List[DiffLoc] = []
    repo = repo.split("/")[-1]

    assert os.path.exists(
        f"{base}/{repo}"
    ), f"[Error] parse_patch: Cannot find {repo} at {base} (Repos should be cloned before running, try first run python dataset/repo_clone.py)"
    proc = subprocess.Popen(
        f"git reset --hard {base_commit}".split(" "),
        cwd=f"{base}/{repo}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()

    try:
        patch_set = PatchSet(patch)
    except Exception as e:
        print(
            f"Warning: returning empty patch, as patch for {instance_id:} is corrupted with msg '{repr(e)}'"
        )
        return ParsedPatch(diff_locs=diff_locs).model_dump_json()
        # raise ValueError(f'{instance_id:}, {patch:}')
    if dbg:
        print(patch_set)

    for file in patch_set:
        if file.is_added_file:
            continue
        if not file.path.endswith(".py"):
            continue
        if not os.path.exists(f"{base}/{repo}/{file.path}"):
            continue
        for hunk in file:
            try:
                src_ranges = get_src_range_from_hunk(hunk, dbg)
                if dbg:
                    print(src_ranges)
                for src_range in src_ranges:
                    diff_nodes = get_diff_nodes_from_file(
                        f"{base}/{repo}/{file.path}", src_range, dbg
                    )
                    if dbg:
                        print(diff_nodes)
                    diff_loc = DiffLoc(
                        file=file.path,
                        diff_nodes=diff_nodes,
                        lineno=src_range.lineno,
                        end_lineno=src_range.end_lineno,
                    )
                    diff_locs.append(diff_loc)
            except:
                raise ValueError(f"{instance_id:}, {hunk:}")
    return ParsedPatch(diff_locs=diff_locs).model_dump_json()


def main():
    """
    All repos should be cloned under BASE before running this script, like:

    (orcar) ~/sandbox/RAGCompiler/OrcarLLM$ ls ../swe_bench_repos/
    astropy  django  flask  matplotlib  pylint  pytest  requests  scikit-learn  seaborn  sphinx  sympy  xarray

    If not cloned, run python dataset/repo_clone.py first
    """
    
    BASE = "../swe_bench_repos"
    OUTPUT_PATH = "./dataset/lite_golden_stats.csv"

    print(f"SWE Bench Repos will be looked under {BASE}")
    print(f"Output will be written to {OUTPUT_PATH}")

    ds = load_dataset("princeton-nlp/SWE-bench_Lite")
    ds_test = pd.DataFrame(ds["test"])
    ds_golden_stats = ds_test[["instance_id", "patch", "repo", "base_commit"]]
    tqdm.pandas()
    ds_golden_stats.insert(
        4,
        "parsed_patch",
        ds_golden_stats.progress_apply(
            lambda row: parse_patch(
                patch=row["patch"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                instance_id=row["instance_id"],
                base=BASE,
            ),
            axis=1,
        ),
    )
    ds_golden_stats.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
