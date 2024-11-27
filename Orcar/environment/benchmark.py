import json
import os
import subprocess
import time
from typing import Any, Dict

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import get_environment_yml, get_requirements

from Orcar.log_utils import get_logger

from .utils import (
    ContainerBash,
    copy_file_to_container,
    get_exit_code,
    run_command_in_container,
)

LONG_TIMEOUT = 500
PATH_TO_REQS = "/root/requirements.txt"
PATH_TO_ENV_YML = "/root/environment.yml"
logger = get_logger(__name__)


def get_repo_dir(repo: str) -> str:
    return repo.replace("/", "__")


def reset_cached_repo(repo_path, base_commit="HEAD"):
    """
    Reset the repo to the base commit.
    """
    cmds = [
        f"git reset --hard {base_commit}",
        f"git submodule update --init --recursive --force",
        f"git submodule deinit -f --all",
        f"rm -rf .git/modules/*",
    ]
    for cmd in cmds:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_path,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()
        out = proc.stdout.read().decode()
        err = proc.stderr.read().decode()
    # Assert git diff has no output
    proc = subprocess.Popen(
        f"git diff",
        cwd=repo_path,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    assert (not out) and (not err), (
        f"Git diff has output after resetting repo {repo_path} to commit {base_commit}:\n"
        "Output {out}\n"
        "Error {err}\n"
    )
    # Check commit ID
    if base_commit != "HEAD":
        result_raw = subprocess.run(
            f"git rev-parse HEAD",
            cwd=repo_path,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        result = result_raw.stdout.strip()
        assert result == base_commit, (
            f"Failed to reset repo {repo_path} to commit {base_commit}:\n"
            f"Current commit: {result}\n"
        )


class BenchmarkEnv:
    def __init__(self, args, ctr_bash: ContainerBash):
        super().__init__()
        self.args = args
        self.ctr_bash = ctr_bash

    def setup(self, inst: Dict[str, Any]):
        logger.info(f"Setting up env for inst {inst['instance_id']}...")
        self.clone_repo(inst)
        self.cache_repo_to_host(inst)
        self.create_conda_env(inst)

    @property
    def cache_dir(self):
        return os.path.expanduser("~/.orcar")

    def copy_to_env(self, contents: str, container_path: str) -> None:
        copy_file_to_container(self.ctr_bash.ctr, contents, container_path)

    def copy_file_from_env(self, container_path: str, host_path: str) -> None:
        cmd = f"docker cp {self.ctr_bash.ctr_name}:/{container_path} {host_path}"
        logger.info(f"Copying file to host: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    def cache_repo_to_host(self, inst: Dict[str, Any]) -> None:
        ctr_name = self.ctr_bash.ctr_name
        repo_name = inst["repo"]
        repo_path = get_repo_dir(repo_name)
        directory_path = self.cache_dir
        os.makedirs(directory_path, exist_ok=True)
        host_path = os.path.join(directory_path, repo_path)
        # check if repo is already cached
        if os.path.exists(host_path):
            logger.info(f"Repo {repo_path} already cached")
        else:
            cmd = f"docker cp {ctr_name}:/{repo_path} {host_path}"
            logger.info(f"Caching repo to host: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        # git checkout to base commit
        base_commit = inst["base_commit"]
        logger.info(f"Checking out {host_path} to base commit: {base_commit}")
        reset_cached_repo(host_path, base_commit)

    def remove_cache_repo(self, repo_name: str) -> None:
        repo_path = get_repo_dir(repo_name)
        host_path = os.path.join(self.cache_dir, repo_path)
        if os.path.exists(host_path):
            os.rmdir(host_path)

    def reset_env_repo(self, repo_dir, commit):
        for cmd in [
            f"cd {repo_dir}",
            f"git reset --hard {commit}",
            f"git submodule update --init --recursive --force",
            f"git submodule deinit -f --all",
            f"rm -rf .git/modules/*",
        ]:
            self.run_with_handle(
                cmd=cmd, err_msg=f"Git failed in {repo_dir} with {cmd}"
            )

    def read_text_file(self, path: str, timeout: int = 5) -> str:
        """
        path: absolute file path (like /tmp/a.log)
        return value: content of text file
        """
        return self.run(cmd=f"cat {path}", timeout=timeout)

    def walk(self, path: str, timeout: int = 5):
        """
        path: absolute file path (like /tmp/a.log)
        return value: type(os.walk(path))
        """
        cmd = "".join(
            [
                "python -c 'import os;",
                "[print(",
                '"{\\"root\\": \\"" + root + "\\", \\"dirs\\": " + repr(dirs) + ", \\"files\\": " + repr(files) + "}"',
                ') for root, dirs, files in os.walk("' + path + "\")]'",
            ]
        )
        output = self.run(cmd=cmd, timeout=timeout).split("\n")[:-1]
        for line in output:
            line_json = json.loads(line.replace("'", '"'))
            yield line_json["root"], line_json["dirs"], line_json["files"]

    def run(self, cmd: str, timeout: int = 5, output_log: bool = False) -> str:
        return run_command_in_container(self.ctr_bash, cmd, timeout, output_log)

    def run_with_handle(
        self, cmd: str, err_msg: str, timeout: int = 5, output_log: bool = False
    ) -> str:
        try:
            output = self.run(cmd, timeout, output_log)
        except Exception:
            raise RuntimeError(err_msg)
        exit_code = get_exit_code(self.ctr_bash, timeout)
        if exit_code != 0:
            raise RuntimeError(f"ErrCode: {exit_code}, {err_msg}")
        return output

    def clone_repo(self, inst: Dict[str, Any]):
        self.run("cd /")
        cur_folders = self.run("ls").split("\n")

        repo = inst["repo"]
        repo_dir = get_repo_dir(repo)
        if repo_dir not in cur_folders:
            logger.info(f"Repo {repo} not found, cloning to /{repo_dir}")
            self.run_with_handle(
                cmd=f"git clone https://github.com/{repo}.git {repo_dir}",
                err_msg=f"Failed to clone repo to {repo_dir}",
                timeout=LONG_TIMEOUT,
                output_log=True,
            )
        self.reset_env_repo(f"/{repo_dir}", inst["environment_setup_commit"])
        self.run(f"cd /")

    def get_cur_conda_envs(self):
        output = self.run("conda env list")
        envs = set([line.split(" ")[0] for line in output.split("\n")])
        envs.discard("")
        envs.discard("#")
        return envs

    def create_conda_env(self, inst: Dict[str, Any]) -> None:
        # Set up environment
        self.run_with_handle(
            "source /root/miniconda3/etc/profile.d/conda.sh",
            err_msg="Failed to source conda",
        )
        inst["repo_dir"] = get_repo_dir(inst["repo"])
        cur_conda_envs = self.get_cur_conda_envs()
        has_runned_container_env_init = False

        t0 = time.perf_counter()
        env_name = inst["repo_dir"] + "__" + inst["version"]

        if env_name in cur_conda_envs:
            return

        if not has_runned_container_env_init:
            has_runned_container_env_init = True
            self.run(f"export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC")
            system = self.run("uname -s").strip().lower()
            arch = self.run("uname -m").strip().lower()
            if system == "linux" and arch == "x86_64":
                self.run_with_handle(
                    "apt update; apt install build-essential -y",
                    err_msg="Failed to install build-essential",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )

        self.run(f"cd /{inst['repo_dir']}")
        logger.info(f"Env {env_name} not found, installing")
        install_configs: dict = MAP_REPO_VERSION_TO_SPECS[inst["repo"]][
            str(inst["version"])
        ]
        packages: str = install_configs.get("packages", "")
        if packages == "requirements.txt":
            # Create conda environment
            self.run_with_handle(
                f"conda create -n {env_name} python={install_configs['python']} -y",
                err_msg="Failed to create conda environment",
                timeout=LONG_TIMEOUT,
                output_log=True,
            )
            # Write reqs to requirements.txt in docker container
            content_reqs = get_requirements(inst)
            self.copy_to_env(content_reqs, PATH_TO_REQS)
            # Create conda environment + install reqs
            self.run_with_handle(
                f"conda activate {env_name}",
                err_msg="Failed to activate conda environment",
            )
            self.run_with_handle(
                f"pip install -r {PATH_TO_REQS}",
                err_msg="Failed to install requirements.txt",
                timeout=LONG_TIMEOUT,
                output_log=True,
            )
            self.run(f"rm {PATH_TO_REQS}")
        elif packages == "environment.yml":
            # Write environment.yml to file
            content_env_yml = get_environment_yml(inst, env_name)
            # Hotfix for
            if not install_configs.get("no_use_env"):
                content_env_yml += f'\n  - python={install_configs["python"]}\n'
            self.copy_to_env(content_env_yml, PATH_TO_ENV_YML)
            if install_configs.get("no_use_env"):
                # Create conda environment
                self.run_with_handle(
                    f"conda create -c conda-forge -n {env_name} python={install_configs['python']} -y",
                    err_msg="Failed to create conda environment",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
                # Install packages
                self.run_with_handle(
                    f"conda env update -f {PATH_TO_ENV_YML}",
                    err_msg="Failed to install environment.yml",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
            else:
                # Create environment + install packages
                self.run_with_handle(
                    f"conda env create --file {PATH_TO_ENV_YML}",
                    err_msg="Failed to create conda environment with environment.yml",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
            self.run(f"rm {PATH_TO_ENV_YML}")
        else:
            python_env = f"python{install_configs['python']}"
            if python_env in cur_conda_envs:
                self.run_with_handle(
                    f"conda create --name {env_name} --clone {python_env}",
                    err_msg="Failed to clone conda environment",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
            else:
                self.run_with_handle(
                    f"conda create -n {env_name} python={install_configs['python']} -y",
                    err_msg="Failed to create conda environment",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
            self.run_with_handle(
                f"conda activate {env_name}",
                err_msg="Failed to activate conda environment",
            )
            if packages.strip():
                self.run_with_handle(
                    f"conda install {packages} -y",
                    err_msg="Failed to install packages",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
        # Install extra pip packages if specified
        if install_configs.get("pip_packages"):
            self.run_with_handle(
                f"source activate {env_name} && pip install {' '.join(install_configs['pip_packages'])}",
                err_msg="Failed to install pip packages",
                timeout=LONG_TIMEOUT,
                output_log=True,
            )

        # Activate environment
        self.run_with_handle(
            f"conda activate {env_name}",
            err_msg="Failed to activate conda environment",
        )

        # Install repo at base commit
        if install_configs.get("pre_install"):
            logger.info("Running pre-install commands...")
            for pre_install_cmd in install_configs["pre_install"]:
                self.run_with_handle(
                    pre_install_cmd,
                    err_msg="Pre-install commands failed to execute successfully",
                    timeout=LONG_TIMEOUT,
                    output_log=True,
                )
        logger.info(f"Installing {inst['repo']} at base commit...")
        if install_configs.get("install"):
            install_cmd = install_configs["install"]
            self.run_with_handle(
                install_cmd,
                err_msg="Install command failed to execute successfully",
                timeout=LONG_TIMEOUT,
                output_log=True,
            )
        if install_configs.get("post_install"):
            logger.info("Running post-install commands...")
            for post_install_cmd in install_configs["post_install"]:
                self.run_with_handle(
                    post_install_cmd,
                    err_msg="Post-install commands failed to execute successfully",
                )

        # Install Tracer
        self.run_with_handle(
            f"conda activate {env_name}",
            err_msg="Failed to activate conda environment",
        )
        self.run_with_handle(
            f"pip install viztracer",
            err_msg="Failed to install viztracer",
            timeout=LONG_TIMEOUT,
            output_log=True,
        )

        logger.info("Installation step took %.2f seconds", time.perf_counter() - t0)
