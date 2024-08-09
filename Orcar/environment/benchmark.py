import datasets
import re
import pandas as pd
import time
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.utils import get_environment_yml, get_requirements
from .utils import ContainerBash, run_command_in_container, copy_file_to_container, get_exit_code, logger

LONG_TIMEOUT = 500
PATH_TO_REQS = "/root/requirements.txt"
PATH_TO_ENV_YML = "/root/environment.yml"

def load_filter_hf_dataset(args) -> datasets.arrow_dataset.Dataset:
    ds = datasets.load_dataset(args.dataset)[args.split]
    return ds.filter(
        input_columns=["instance_id"],
        function=lambda x: bool(re.match(args.filter_instance, x)),
    )

def get_repo_dir(repo: str) -> str:
    return repo.replace("/", "__")

class BenchMarkEnv:
    def __init__(self, args, ctr_bash: ContainerBash, ds: datasets.arrow_dataset.Dataset):
        super().__init__()
        self.args = args
        self.ctr_bash = ctr_bash
        self.ds = pd.DataFrame(ds)
        self.clone_repos()
        self.create_conda_envs()
        
        

    def run(self, cmd: str, timeout: int=5, output_log: bool = False) -> str:
        return run_command_in_container(self.ctr_bash, cmd, timeout, output_log)

    def run_with_handle(self, cmd: str, err_msg: str, timeout: int=5, output_log: bool = False) -> str:
        try:
            output = self.run(cmd, timeout, output_log)
        except:
            raise RuntimeError(err_msg)
        exit_code = get_exit_code(self.ctr_bash, timeout)
        if (exit_code != 0):
            raise RuntimeError(f"ErrCode: {exit_code}, {err_msg}")
        return output
    
    def clone_repos(self):
        self.run("cd /")
        cur_folders = self.run("ls").split("\n")
        repo_list = self.ds.drop_duplicates('repo')
        for _, row in repo_list.iterrows():
            repo = row['repo']
            repo_dir = get_repo_dir(repo)
            if repo_dir not in cur_folders:
                logger.info(f"Repo {repo} not found, cloning to /{repo_dir}")
                self.run_with_handle(cmd=f"git clone https://github.com/{repo}.git {repo_dir}", err_msg=f"Failed to clone repo to {repo_dir}", timeout=LONG_TIMEOUT, output_log=True)
            for cmd in [
                f"cd /{repo_dir}",
                "git status",
                f"git checkout {row['environment_setup_commit']}",
                "cd /"
                ]:
                self.run_with_handle(cmd=cmd, err_msg=f"Git failed in {repo_dir}")

    def get_cur_conda_envs(self):
        output = self.run("conda env list")
        envs = set([line.split(' ')[0] for line in output.split('\n')])
        envs.discard('')
        envs.discard('#')
        return envs

    def create_conda_envs(self):
        # Set up environment
        self.run_with_handle(
            "source /root/miniconda3/etc/profile.d/conda.sh",
            err_msg="Failed to source conda",
        )

        conda_envs = self.ds.drop_duplicates(['repo', 'version'])
        conda_envs.insert(0, 'repo_dir', conda_envs.repo.apply(get_repo_dir))
        conda_envs.insert(0, 'conda_env_name', conda_envs.repo_dir + '__' + conda_envs.version)
        cur_conda_envs = self.get_cur_conda_envs()
        for _, row in conda_envs.iterrows():
            t0 = time.perf_counter()
            record = dict(row)
            env_name = record['conda_env_name']
            
            if env_name in cur_conda_envs:
                continue
            self.run(f"cd /{record['repo_dir']}")
            logger.info(f"Env {env_name} not found, installing")
            install_configs: dict = MAP_REPO_VERSION_TO_SPECS[record['repo']][str(record['version'])]
            packages: str = install_configs.get("packages", "")
            if packages == "requirements.txt":
                # Create conda environment
                self.run_with_handle(
                    f"conda create -n {env_name} python={install_configs['python']} -y",
                    err_msg="Failed to create conda environment",
                    timeout=LONG_TIMEOUT, output_log=True
                )
                logger.debug("Created conda environment")
                # Write reqs to requirements.txt in docker container
                content_reqs = get_requirements(record)
                copy_file_to_container(self.ctr_bash.ctr, content_reqs, PATH_TO_REQS)
                # Create conda environment + install reqs
                self.run_with_handle(
                    f"conda activate {env_name}",
                    err_msg="Failed to activate conda environment",
                )
                self.run_with_handle(
                    f"pip install -r {PATH_TO_REQS}",
                    err_msg="Failed to install requirements.txt",
                    timeout=LONG_TIMEOUT, output_log=True
                )
                logger.debug("Installed requirements from requirements.txt")
                self.run(f"rm {PATH_TO_REQS}")
            elif packages == "environment.yml":
                # Write environment.yml to file
                content_env_yml = get_environment_yml(record, env_name)
                # Hotfix for
                if not install_configs.get("no_use_env"):
                    content_env_yml += f'\n  - python={install_configs["python"]}\n'
                copy_file_to_container(self.ctr_bash.ctr, content_env_yml, PATH_TO_ENV_YML)
                if install_configs.get("no_use_env"):
                    # Create conda environment
                    self.run_with_handle(
                        f"conda create -c conda-forge -n {env_name} python={install_configs['python']} -y",
                        err_msg="Failed to create conda environment",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                    logger.debug("Created conda environment")
                    # Install packages
                    self.run_with_handle(
                        f"conda env update -f {PATH_TO_ENV_YML}",
                        err_msg="Failed to install environment.yml",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                    logger.debug("Installed packages from environment.yml")
                else:
                    # Create environment + install packages
                    self.run_with_handle(
                        f"conda env create --file {PATH_TO_ENV_YML}",
                        err_msg="Failed to create conda environment with environment.yml",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                    logger.debug("Created conda environment with environment.yml")
                self.run(f"rm {PATH_TO_ENV_YML}")
            else:
                python_env = f"python{install_configs['python']}"
                if python_env in cur_conda_envs:
                    self.run_with_handle(
                        f"conda create --name {env_name} --clone {python_env}",
                        err_msg="Failed to clone conda environment",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                    logger.debug("Cloned python conda environment")
                else:
                    logger.debug(f"Could not find {python_env}, creating new environment")
                    self.run_with_handle(
                        f"conda create -n {env_name} python={install_configs['python']} -y",
                        err_msg="Failed to create conda environment",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                self.run_with_handle(
                    f"conda activate {env_name}",
                    err_msg="Failed to activate conda environment",
                )
                if packages.strip():
                    self.run_with_handle(
                        f"conda install {packages} -y",
                        err_msg="Failed to install packages",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                    logger.debug("Installed conda packages")
            # Install extra pip packages if specified
            if install_configs.get("pip_packages"):
                self.run_with_handle(
                    f"source activate {env_name} && pip install {' '.join(install_configs['pip_packages'])}",
                    err_msg="Failed to install pip packages",
                    timeout=LONG_TIMEOUT, output_log=True
                )
                logger.debug("Installed extra pip dependencies")

            # Activate environment
            self.run_with_handle(f"conda activate {env_name}", err_msg="Failed to activate conda environment")

            # Install repo at base commit
            if install_configs.get("pre_install"):
                logger.info("Running pre-install commands...")
                for pre_install_cmd in install_configs["pre_install"]:
                    self.run_with_handle(
                        pre_install_cmd,
                        err_msg="Pre-install commands failed to execute successfully",
                        timeout=LONG_TIMEOUT, output_log=True
                    )
                logger.debug("Ran pre-install commands")
            logger.info(f"Installing {record['repo']} at base commit...")
            if install_configs.get("install"):
                install_cmd = install_configs["install"]
                self.run_with_handle(
                    install_cmd,
                    err_msg="Install command failed to execute successfully",
                    timeout=LONG_TIMEOUT, output_log=True
                )
                logger.debug("Ran install command")
            if install_configs.get("post_install"):
                logger.info("Running post-install commands...")
                for post_install_cmd in install_configs["post_install"]:
                    self.run_with_handle(
                        post_install_cmd,
                        err_msg="Post-install commands failed to execute successfully",
                    )
                logger.debug("Ran post-install commands")

            logger.info("Installation step took %.2f seconds", time.perf_counter() - t0)



    

