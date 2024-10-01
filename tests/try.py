from abc import abstractmethod
from typing import List

from llama_index.core.bridge.pydantic import BaseModel


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""


class SearchActionStep(BaseReasoningStep):
    """Search action reasoning step."""

    action: str
    action_input: List

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Search Action: {self.action}\n"
            f"Search Action Input: {self.action_input}"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


api_calls = [
    "api_call_1(arg1, arg2)",
    "api_call_2(arg1, arg2)",
    # Add more API calls as needed
]

# api_calls contain api_call_1 and args
# api_calls = [api_call_1(args), api_call_2(args), ...]
# in SearchActionStep, action is api_call_1, action_input is args
for api_call in api_calls:
    action, action_input = api_call.split("(")
    action_input = action_input[:-1]
    a = SearchActionStep(action=action, action_input=action_input.split(", "))
    print(a)
