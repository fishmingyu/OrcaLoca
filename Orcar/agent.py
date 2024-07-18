from typing import Any, Iterator, Sequence, Optional, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
)

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph, START
from .planner import Planner
from .joiner import Joiner

from langchain_core.runnables import (
    Runnable
)


def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"


class OrcarAgent:
    """Orcar Agent.
    """
    def __init__(
        self,
        tools: Sequence[BaseTool],
    ) -> None:
        self.tools = tools
        self.plan_llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.planner = Planner(tools, self.plan_llm)
        self.join_llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.joiner = Joiner(self.join_llm)
        self.chain = self.build_graph()

    def build_graph(self):
        graph_builder = MessageGraph()

        # 1.  Define vertices
        # We defined plan_and_schedule above already
        # Assign each node to a state variable to update
        graph_builder.add_node("plan_and_schedule", self.planner)
        graph_builder.add_node("join", self.joiner)

        ## Define edges
        graph_builder.add_edge("plan_and_schedule", "join")

        ### This condition determines looping logic

        graph_builder.add_conditional_edges(
            "join",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
        )
        graph_builder.add_edge(START, "plan_and_schedule")
        chain = graph_builder.compile()
        return chain
    
    def invoke(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        return self.chain.invoke(messages)
    
    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessage]:
        return self.chain.stream(messages)