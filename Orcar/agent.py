from .instructor import ReActAgent
from typing import Any, Dict, List, Sequence, Tuple


from llama_index.llms.openai import OpenAI

import time

from .tools import get_llm_compiler_executer, create_tool_list
from .environment.utils import ContainerBash

# construct the Orcar agent

class OrcarAgent:
    """Orcar agent worker."""
    def __init__(self, args, cfg, enable_jit: bool = True, ctr_bash: ContainerBash | None = None):
        super().__init__()
        self.ctr_bash = ctr_bash
        self.llm = OpenAI(model=args.model, api_key=cfg['OPENAI_API_KEY'], api_base=cfg['OPENAI_API_BASE_URL'])
        self.instructor = self.create_instructor(self.llm, enable_jit)
        

    def chat(self, text: str) -> str:
        """Chat with the agent."""
        response = self.instructor.chat(text)
        return response

    def run(self, text: str) -> str:
        """Run the agent."""
        response = self.chat(text)
        return response

    def __call__(self, text: str) -> str:
        """Call the agent."""
        return self.run(text)
    
    def create_instructor(self, llm: OpenAI, enable_jit) -> ReActAgent:
        tool_list = create_tool_list(self.ctr_bash)
        if enable_jit:
            llm_compiler_tool = get_llm_compiler_executer(llm, tool_list)
            return ReActAgent.from_tools(llm=llm, tools=[llm_compiler_tool], verbose=True)
        else:
            return ReActAgent.from_tools(llm=llm, tools=tool_list, verbose=True)