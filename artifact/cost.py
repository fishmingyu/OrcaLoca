"""
log folder structure:

log/
├── astropy__astropy-12907
│   ├── ...
│   ├── orcar_total.log
├── astropy__astropy-6938
│   ├── ...
│   ├── orcar_total.log
└── sympy__sympy-12419
    ├── ...
    └── orcar_total.log

Key line format in orcar_total.log:
[2024-12-29 00:12:43,265 - Orcar.xxx_agent - INFO] Total cnt                : in 123960 tokens, out   4493 tokens

The function of this script is to calculate the cost of each instance based on the key line in the log file.
"""

import os
import re
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--log_dir",
        default="./log",
        type=str,
        help="Name of directory of log.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    log_dir: str = args.log_dir
    # Get all log folders
    log_folders = [os.path.join(log_dir, f) for f in os.listdir(log_dir)]
    log_files = []
    for folder in log_folders:
        # Get all log files named orcar_total.log
        log_files.extend(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f == "orcar_total.log"
            ]
        )

    # Calculate sum of in and out tokens for all instances
    in_tokens = 0
    out_tokens = 0
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f.readlines():
                # Extract in and out tokens
                match = re.search(
                    r"Total cnt +: in +(\d+) tokens, out +(\d+) tokens", line
                )
                if match:
                    print("Found line: ", line.strip())
                    in_tokens += int(match.group(1))
                    out_tokens += int(match.group(2))

    in_token_cost = 3 / 1000000
    out_token_cost = 15 / 1000000
    # Calculate cost
    in_cost = in_tokens * in_token_cost
    out_cost = out_tokens * out_token_cost
    total_cost = in_cost + out_cost
    print(f"Found insts: {len(log_files)}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Cost per instance: ${total_cost / len(log_files):.2f}")


if __name__ == "__main__":
    main()
