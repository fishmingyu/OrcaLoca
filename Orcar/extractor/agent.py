from typing import List, Dict
import pandas as pd
from pathlib import PurePosixPath, PureWindowsPath, PurePath

from llama_index.llms.openai import OpenAI

from Orcar.environment.benchmark import BenchmarkEnv, get_repo_dir
from Orcar.environment.utils import get_logger

from .schema import *

logger = get_logger("extractor")


class ExtractorAgent:
    def __init__(self, cfg, benchmark_env: BenchmarkEnv):
        self.env = benchmark_env
        self.llm = OpenAI(
            model=self.env.args.model,
            api_key=cfg["OPENAI_API_KEY"],
            api_base=cfg["OPENAI_API_BASE_URL"],
        )
        self.result: Dict[str, str] = dict()
        self.extractor_program = get_extractor_function(llm=self.llm)
        self.reproducer_program = get_reproducer_function(llm=self.llm)

    def run(self):
        for _, inst in self.env.ds.iterrows():
            self.result[inst["instance_id"]] = self.run_inst(inst)

    def run_inst(self, inst: pd.Series):
        repo_dir = get_repo_dir(inst["repo"])
        for cmd in [
            f"cd /{repo_dir}",
            f"conda activate {repo_dir + '__' + inst['version']}",
            f"git reset --hard {inst['base_commit']}",
        ]:
            self.env.run_with_handle(
                cmd=cmd, err_msg=f"Inst {inst['instance_id']} failed at {cmd=}"
            )
        logger.info("LLM extracting from issue text...")
        output: RawIssueInfo = self.extractor_program(
            issue_description=inst["problem_statement"]
        )
        logger.debug(output.summary)
        logger.debug(output.issue_reproducer)
        logger.debug(output.related_code_snippets)

        parsed_codeinfo = self.parse_path_in_code_info(
            output.related_code_snippets, inst
        )
        logger.debug(parsed_codeinfo)
        reproducer_output = self.reproduce_issue(output.issue_reproducer, inst)
        logger.debug(reproducer_output)
        output = IssueInfo(
            summary=output.summary,
            issue_reproducer_info=reproducer_output,
            related_code_snippets=parsed_codeinfo,
        ).model_dump_json()
        logger.debug(output)
        return output

    def reproduce_issue(self, issue_reproducer: str, inst: pd.Series) -> ReproducerInfo:
        reproducer_path = (
            f"/tmp/reproducer_{get_repo_dir(inst['repo'])}__{inst['version']}.py"
        )
        self.env.copy_to_env(issue_reproducer, reproducer_path)
        logger.info("Running reproducer...")
        log = self.env.run(f"python {reproducer_path}", output_log=True)
        reproduce_history = (
            f"<reproducer_snippet>{issue_reproducer}</reproducer_snippet>\n"
            + f"<execution_log>{log}</execution_log>"
        )
        logger.debug(reproduce_history)
        logger.info("LLM judging the reproduction...")
        return self.reproducer_program(
            issue_description=inst["problem_statement"],
            reproduce_history=reproduce_history,
        )

    def parse_path_in_code_info(
        self, related_code_snippets: List[CodeLocationInfo], inst: pd.Series
    ) -> List[CodeLocationInfo]:

        def cut_since_last_sensitive(target, sensitive):
            # Find the last occurrence of any element from sensitive in target
            for i in range(len(target) - 1, -1, -1):
                if target[i] in sensitive:
                    target = target[i:]
                    break
            return target

        def detect_path_fs(path_str: str):
            # Distinguish file system: Windows or Posix
            path_posix_expression = PurePosixPath(path_str)
            path_windows_expression = PureWindowsPath(path_str)
            path_ret = path_posix_expression
            if (
                len(path_posix_expression.parts) < len(path_windows_expression.parts)
                or path_windows_expression.is_absolute()
            ):
                path_ret = path_windows_expression
            return path_ret

        processed_code_info_list: List[CodeLocationInfo] = []
        sensitive_list = ["tests"]
        repo_folder = inst["repo"].split("/")[-1]
        repo_root = "/" + get_repo_dir(inst["repo"])
        if inst["repo"] == "scikit-learn/scikit-learn":
            repo_folder = "sklearn"
        sensitive_list += [repo_folder]

        for code_info in related_code_snippets:
            path = detect_path_fs(code_info.file_path)

            if not any([p in path.parts for p in sensitive_list]):
                continue
            relative_path_suffix = PurePath(
                *cut_since_last_sensitive(path.parts, sensitive_list)
            )

            if path.is_absolute() or path.parts[0] == "~":
                find_output = self.env.run(f"find {repo_root} -name {path.parts[-1]}")
                candidates = find_output.split("\n")
                output_paths = list(
                    filter(lambda x: x.endswith(str(relative_path_suffix)), candidates)
                )
                for x in output_paths:
                    processed_code_info = code_info.copy(deep=True)
                    processed_code_info.file_path = x
                    processed_code_info_list.append(processed_code_info)
            else:
                processed_code_info_list.append(code_info)
        return processed_code_info_list
