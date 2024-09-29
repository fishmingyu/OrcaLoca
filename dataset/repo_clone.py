import os
import subprocess

import pandas as pd
from datasets import load_dataset


def main():
    """
    Clone all repos under give path
    """
    PATH = "../swe_bench_repos"
    assert os.path.exists(
        PATH
    ), f"[Error] parse_patch: Cannot find clone target folder {PATH}"

    ds = load_dataset("princeton-nlp/SWE-bench_Lite")
    ds_test = pd.DataFrame(ds["test"])
    repos = set(ds_test.repo.to_list())
    for repo in repos:
        repo_folder_name = repo.split("/")[-1]
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo}.git",
                f"{PATH}/{repo_folder_name}",
            ]
        )


if __name__ == "__main__":
    main()
