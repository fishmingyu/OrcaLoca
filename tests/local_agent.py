import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from Orcar import OrcarAgent
import subprocess
import os

# tool bindings
# orcar_agent = OrcarAgent()
# response = orcar_agent.chat("What is (121 * 3) + 42? Calculate step by step. Make a new directory named tests, and then write the calculated result into a new file named 'result.txt' in this directory.")

# print(response)


from Orcar.environment.env import EnvironmentSetup

image_name = "orcar/x86_64:latest"

config_path = "../key.cfg"
absolute_path = os.path.abspath(config_path)

container = EnvironmentSetup(image_name=image_name, config_path=absolute_path)


#  1. copy directory
#  2. install requirements
#  3. run agent (CLI)