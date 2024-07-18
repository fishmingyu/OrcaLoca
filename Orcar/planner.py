from typing import Sequence, Optional, Dict, List

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.runnables import (
    chain as as_runnable,
    RunnableBranch,
    Runnable
)
from langchain.agents.agent import RunnableAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from .output_parser import LLMCompilerPlanParser, Task
from .task_fetching_unit import schedule_tasks
from langgraph.graph import END, MessageGraph, START
import itertools


class Planner:
    """LLMCompiler Planner

    Source Repo (paper linked): https://github.com/SqueezeAILab/LLMCompiler?tab=readme-ov-file

    """
    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: BaseChatModel,
    ) -> None:
        self.planner_prompt = hub.pull("wfh/llm-compiler")
        self.llm = llm
        self.tools = tools
        self.planner = self.create_planner(llm, tools, self.planner_prompt)

    def create_planner(
        self, llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
    ):
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.description}\n"
            for i, tool in enumerate(
                tools
            )  # +1 to offset the 0 starting index, we want it count normally from 1.
        )
        planner_prompt = base_prompt.partial(
            replan="",
            num_tools=len(tools)
            + 1,  # Add one because we're adding the join() tool at the end.
            tool_descriptions=tool_descriptions,
        )
        replanner_prompt = base_prompt.partial(
            replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
            " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
            num_tools=len(tools) + 1,
            tool_descriptions=tool_descriptions,
        )

        def should_replan(state: list):
            # Context is passed as a system message
            return isinstance(state[-1], SystemMessage)

        def wrap_messages(state: list):
            return {"messages": state}

        def wrap_and_get_last_index(state: list):
            next_task = 0
            for message in state[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs["idx"] + 1
                    break
            state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
            return {"messages": state}

        return (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | planner_prompt,
            )
            | llm
            | LLMCompilerPlanParser(tools=tools)
        )

    def __call__(self, messages: List[BaseMessage], config):
        tasks = self.planner.stream(messages, config)
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        scheduled_tasks = schedule_tasks.invoke(
            {
                "messages": messages,
                "tasks": tasks,
            },
            config,
        )
        return scheduled_tasks