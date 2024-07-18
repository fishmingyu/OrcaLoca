from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from typing import Union, List, Sequence
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import (
    chain as as_runnable,
    Runnable
)
from langchain_core.tools import BaseTool


class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )

class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


class Joiner:
    """LLMCompiler Joiner

    Source Repo (paper linked): https://github.com/SqueezeAILab/LLMCompiler?tab=readme-ov-file

    """
    def __init__(
        self,
        llm: BaseChatModel,
    ) -> None:
        self.planner_prompt = hub.pull("wfh/llm-compiler")
        self.llm = llm

    def _parse_joiner_output(self, decision: JoinOutputs) -> List[BaseMessage]:
        response = [AIMessage(content=f"Thought: {decision.thought}")]
        if isinstance(decision.action, Replan):
            return response + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
        else:
            return response + [AIMessage(content=decision.action.response)]

    def select_recent_messages(self, messages: list) -> dict:
        selected = []
        for msg in messages[::-1]:
            selected.append(msg)
            if isinstance(msg, HumanMessage):
                break
        return {"messages": selected[::-1]}

    
    def __call__(self, messages: list):
        joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
            examples=""
        )  # You can optionally add examples
        runnable = joiner_prompt | self.llm.with_structured_output(JoinOutputs)
        joiner = self.select_recent_messages | runnable | self._parse_joiner_output
        return joiner