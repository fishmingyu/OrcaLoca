from .utils import *
import os

class EnvironmentSetup:
    def __init__(self, image_name: str, config):
        self.image_name = image_name
        self.config = config
        self.setup()

    def setup(self):
        container = get_container(ctr_name="orcar-agent", image_name='sweagent/swe-agent:latest')[0]
        command = "echo 'Hello, World!'"
        run_command_in_container(container, command)
        setup_key_command = f"echo 'export OPENAI_API_KEY={self.config['OPENAI_API_KEY']}' >> ~/.bashrc"
        run_command_in_container(container, setup_key_command)
        run_agent_build_command = "source ~/.bashrc"
        run_command_in_container(container, run_agent_build_command)
        enter_conda_env_command = "source activate test"
        run_command_in_container(container, enter_conda_env_command)
        