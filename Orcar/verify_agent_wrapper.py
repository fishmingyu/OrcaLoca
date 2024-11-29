from typing import Any, Dict

from llama_index.core.llms.llm import LLM

from .environment.benchmark import BenchmarkEnv, get_repo_dir
from .log_utils import get_logger
from .types import VerifyOutput

logger = get_logger(__name__)


class VerifyAgentWrapper:
    def __init__(self, llm: LLM, env: BenchmarkEnv, instance: Dict[str, Any]):
        self.llm = llm
        self.env = env
        self.inst = instance
        self.inst_id = self.inst["instance_id"]
        self.env_repo_dir = f"/{get_repo_dir(repo=self.inst['repo'])}"
        # TODO: Implement VerifyAgent
        # self.verify_agent = VerifyAgent(llm=llm, env=env, verbose=True)
        self.is_reproduce_pass = False
        self.verify_snippet = ""

        self.verify_snippet_dir = "/tmp/orcar_verify_snippet"
        self.env.run(f"mkdir -p {self.verify_snippet_dir}")
        self.verify_snippet_path: str | None = None

    def init_verify_script(
        self, is_reproduce_pass: bool, reproduce_snippet: str
    ) -> None:
        self.is_reproduce_pass = is_reproduce_pass
        self.verify_snippet = reproduce_snippet
        # TODO: Confirm reproduce snippet is not failing because of viztracer
        # TODO: improve verify snippet with LLM
        if self.is_reproduce_pass and self.verify_snippet:
            self.verify_snippet_path = (
                f"{self.verify_snippet_dir}/verify_{self.inst_id}.py"
            )
            self.env.run(f"rm {self.verify_snippet_path}")
            self.env.copy_to_env(self.verify_snippet, self.verify_snippet_path)
            logger.info(f"Verify snippet is ready at {self.verify_snippet_path}")
            logger.info(f"Snippet content: \n{self.verify_snippet}")

    def verify_patch(self, patch_to_verify: str) -> VerifyOutput:
        if not self.verify_snippet_path:
            is_error = True
            error_msg = "Verify snippet is not available. Skip verifying."
            logger.info(error_msg)
            verify_log = ""
            return VerifyOutput(
                is_error=is_error, error_msg=error_msg, verify_log=verify_log
            )
        self.env.run(f"cd {self.env_repo_dir}")
        self.env.reset_env_repo(
            repo_dir=self.env_repo_dir, commit=self.inst["base_commit"]
        )
        if patch_to_verify:
            logger.info(f"Verify Agent got patch: \n{patch_to_verify}")
            patch_path = f"{self.verify_snippet_dir}/patch_{self.inst_id}.patch"
            self.env.copy_to_env(patch_to_verify, patch_path)
            apply_log, exit_code = self.env.run_with_exit_code(
                f"git apply {patch_path}"
            )
            if exit_code != 0:
                is_error = True
                error_msg = f"Failed to apply patch: {apply_log}"
                logger.info(error_msg)
                verify_log = ""
                return VerifyOutput(
                    is_error=is_error, error_msg=error_msg, verify_log=verify_log
                )
        else:
            logger.info("No patch to verify. Skip applying")
        verify_log, exit_code = self.env.run_with_exit_code(
            f"python {self.verify_snippet_path}"
        )

        logger.info(f"Verify log:\n{verify_log}")
        is_error = exit_code != 0
        error_msg = "Verify failed" if is_error else ""
        if error_msg:
            logger.info(error_msg)
        if patch_to_verify:
            self.env.run(f"rm {patch_path}")
        self.env.reset_env_repo(
            repo_dir=self.env_repo_dir, commit=self.inst["base_commit"]
        )
        return VerifyOutput(
            is_error=is_error, error_msg=error_msg, verify_log=verify_log
        )
