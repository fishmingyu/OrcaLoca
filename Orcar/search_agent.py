"""
A search agent. Process raw response into json format.
"""

import uuid
import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput
from openai.types.chat import ChatCompletionMessageToolCall
from .types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.prompts.mixin import PromptMixinType, PromptDictType

import logging
from .environment.utils import get_logger
from .search import SearchManager
from .formatter import SearchChatFormatter
from .output_parser import SearchOutputParser
from .types import SearchStep

logger = get_logger("search_agent")
dispatcher = get_dispatcher(__name__)

class SearchWorker(BaseAgentWorker):
    """OpenAI Agent worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        max_iterations: int = 10,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        self._llm = llm
        self.callback_manager = callback_manager or llm.callback_manager
        self._max_iterations = max_iterations
        self._search_formatter = search_formatter or SearchChatFormatter()
        self._output_parser = output_parser or SearchOutputParser()
        self._verbose = verbose

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        max_iterations: int = 10,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "SearchWorker":
        """Convenience constructor method from set of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            SearchWorker
        """
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            search_formatter=search_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        sys_header = self._search_formatter.system_header
        return {"system_prompt": PromptTemplate(sys_header)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "system_prompt" in prompts:
            sys_prompt = cast(PromptTemplate, prompts["system_prompt"])
            self._search_formatter.system_header = sys_prompt.template

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        current_search: List[SearchStep] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "sources": sources,
            "current_search": current_search,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"is_first": True},
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        return [t for t in self._get_tools(input)]

    def _extract_search_step(
        self, output: ChatResponse
    ) -> Tuple[str, List[SearchStep], bool]:
        """Extract search step."""
        # parse the output
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        current_search = []
        try:
            search_step = self._output_parser.parse(message_content)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        current_search.append(search_step)
        if self._verbose:
            logger.info(f"Search step: {search_step.get_content()}")
        logger.info(f"Is done: {search_step.is_done}")
        if search_step.is_done:
            return message_content, current_search, True
        return message_content, current_search, False

    def _process_search(
        self,
        task: Task,
        tools: Sequence[BaseTool],
        output: ChatResponse,
    ) -> Tuple[List[SearchStep], bool]:
        tools_dict: Dict[str, BaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        # try to call the tools
        search_steps = []
        is_done = False
        try:
            search_output, search_steps, is_done = self._extract_search_step(output)
        except Exception as exc:
            logger.error(f"Error processing search: {exc}")
            search_output = output.message.content
            is_done = True
        # add search steps to task state
        task.extra_state["current_search"].extend(search_steps)
        return search_steps, is_done

    def _get_response(
        self,
        current_search: List[BaseReasoningStep],
        sources: List[ToolOutput],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_search) == 0:
            raise ValueError("No searching steps were taken.")
        elif len(current_search) == self._max_iterations:
            raise ValueError("Reached max iterations.")
        response_str = current_search[-1].get_content()

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_str, sources=sources)

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)
        chat_history = task.memory.get_all()
        # add task input to chat history
        chat_history.append(ChatMessage(content=task.input, role=MessageRole.USER))
        input_chat = self._search_formatter.format(
            tools,
            chat_history=chat_history + task.extra_state["new_memory"].get_all(),
            current_search=task.extra_state["current_search"],
        )

        # send prompt
        chat_response = self._llm.chat(input_chat)
        logger.info(f"Chat response: {chat_response}")
        # given prompt outputs, call search tools
        searching_steps, is_done = self._process_search(
            task, tools, output=chat_response
        )
        task.extra_state["current_search"].extend(searching_steps)
        agent_response = self._get_response(
            task.extra_state["current_search"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)
    

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, task)
    
    def arun_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        return super().arun_step(step, task, **kwargs)
    
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        pass 

    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        pass
    
    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(
            task.memory.get_all() + task.extra_state["new_memory"].get_all()
        )
        # reset new memory
        task.extra_state["new_memory"].reset()

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        # TODO: make this abstractmethod (right now will break some agent impls)
        self.callback_manager = callback_manager


class SearchAgent(AgentRunner):
    """ReAct agent.

    Subclasses AgentRunner with a ReActAgentWorker.

    For the legacy implementation see:
    ```python
    from llama_index.core.agent.legacy.react.base import ReActAgent
    ```

    """

    def __init__(
        self,
        llm: LLM,
        repo_path: str = "",
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        max_iterations: int = 10,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager

        self._search_manager = SearchManager(repo_path=repo_path)
        self._tools = self._setup_tools()

        step_engine = SearchWorker.from_tools(
            tools=self._tools,
            llm=llm,
            max_iterations=max_iterations,
            search_formatter=search_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        super().__init__(
            step_engine,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
        )
    def _setup_tools(self) -> List[BaseTool]:
        """Set up tools."""
        tools = []
        # tools in SearchManager
        search_tool = FunctionTool.from_defaults(self._search_manager.search_func)
        tools.append(search_tool)
        return tools

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"agent_worker": self.agent_worker}