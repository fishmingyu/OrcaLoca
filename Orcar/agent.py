from .instructor import ReActAgent
from typing import Any, Dict, List, Sequence, Tuple


from llama_index.llms.openai import OpenAI

import subprocess


from .tools import get_llm_compiler_executer

# construct the Orcar agent

class OrcarAgent:
    """Orcar agent worker."""
    def __init__(self, args, cfg, ctr_name = ""):
        super().__init__()
        self.ctr_name = ctr_name
        self.llm = OpenAI(model=args.model, api_key=cfg['OPENAI_API_KEY'], api_base=cfg['OPENAI_API_BASE_URL'])
        self.instructor = self.create_instructor(self.llm)
        

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
    
    def create_instructor(self, llm: OpenAI) -> ReActAgent:
        llm_compiler_tool = get_llm_compiler_executer(llm, self.ctr_name)
        return ReActAgent.from_tools(llm=llm, tools=[llm_compiler_tool], verbose=True)