"""
A search agent. Process raw response into json format.
"""

import uuid
import json
import queue
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
    SearchResult
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
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent


import logging
from .environment.utils import get_logger
from .search import SearchManager
from .formatter import SearchChatFormatter
from .output_parser import SearchOutputParser
from .types import SearchActionStep, SearchObservationStep

logger = get_logger("search_agent")
dispatcher = get_dispatcher(__name__)


def add_user_step_to_memory(
    step: TaskStep,
    memory: BaseMemory,
) -> None:
    """Add user step to memory."""
    if "is_first" in step.step_state and step.step_state["is_first"]:
        # add to new memory
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        step.step_state["is_first"] = False
    else:   # conclusion
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        # logger.info(f"Add user input to memory: {step.input}")

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
        next_step: str = ""
        next_step_input: str = ""
        sources: List[ToolOutput] = []
        search_queue: queue.Queue = queue.Queue()
        action_history: List[SearchActionStep] = []
        current_search: List[SearchResult] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "next_step": next_step,
            "next_step_input": next_step_input,
            "sources": sources,
            "search_queue": search_queue,
            "action_history": action_history,
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

    def _extract_exploring_step(
        self, output: ChatResponse
    ) -> Tuple[str, List[BaseReasoningStep]]:
        """Extract search step."""
        # parse the output
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        try:
            obseravtion, explore_step = self._output_parser.parse_explore(message_content)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        return obseravtion, explore_step
        
    def _assign_next_step(self, is_complete: bool) -> str:
        """Assign next step."""
        if is_complete:
            return "conclusion"
        else:
            return "explore"

    def _process_search(
        self,
        task: Task,
        tools: Sequence[BaseTool],
        search_step: SearchActionStep,
    ) -> SearchResult:
        tools_dict: Dict[str, BaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        # try to call the tools
        if search_step.action in tools_dict:
            tool = tools_dict[search_step.action]
            with self.callback_manager.event(
                CBEventType.FUNCTION_CALL,
                payload={
                    EventPayload.FUNCTION_CALL: search_step.action_input,
                    EventPayload.TOOL: tool.metadata,
                },
            ) as event:
                try:
                    dispatcher.event(
                        AgentToolCallEvent(
                            arguments=json.dumps({**search_step.action_input}),
                            tool=tool.metadata,
                        )
                    )
                    tool_output = tool.call(**search_step.action_input)
                except Exception as e:
                    tool_output = ToolOutput(
                        content=f"Error: {e!s}",
                        tool_name=tool.metadata.name,
                        raw_input={"kwargs": search_step.action_input},
                        raw_output=e,
                        is_error=True,
                    )
                event.on_end(
                    payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)}
                )
        else:
            tool_output = ToolOutput(
                content=f"Error: Tool {search_step.action} not found.",
                tool_name=search_step.action,
                raw_input={"kwargs": search_step.action_input},
                raw_output=None,
                is_error=True,
            )
        search_result = SearchResult(search_action=search_step.action, search_input=search_step.action_input, search_content=tool_output.content)
        task.extra_state["sources"].append(tool_output)

        return search_result 
    
    def _concat_search_results(self, search_results: List[SearchResult]) -> str:
        """Join and Concatenate search results."""
        search_results_str = ""
        for search_result in search_results:
            search_results_str += search_result.get_content() + "\n"
        return search_results_str
    
    def _get_response(
        self,
        current_res: List[BaseReasoningStep],
        sources: List[ToolOutput],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_res) == 0:
            raise ValueError("No searching steps were taken.")
        elif len(current_res) == self._max_iterations:
            raise ValueError("Reached max iterations.")
        response_str = current_res[-1].get_content()

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_str, sources=sources)

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, next_step: str, next_step_input: str, is_done: bool = False
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        elif next_step == "conclusion":
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input="Now let's come to a conclusion. \n"
                    , # this step is conclusion
                )
            ]
        elif next_step == "explore":
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input="Please provide observation feedback and new_search_actions on the search results below. \n"
                    + next_step_input
                    , # this step is observation
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
        if step.input is not None:
            add_user_step_to_memory(
                step,
                task.extra_state["new_memory"],
            )
        tools = self.get_tools(task.input)
        # add task input to chat history
        input_chat = self._search_formatter.format(
            tools,
            chat_history=task.memory.get(input=task.input) + task.extra_state["new_memory"].get_all(),
            current_search=task.extra_state["current_search"],
        )
        # send prompt
        chat_response = self._llm.chat(input_chat, response_format={"type": "json_object"})
        logger.info(f"Chat response: {chat_response}")
        if task.extra_state["next_step"] is "conclusion":
            # convert the chat response to str
            chat_response = str(chat_response)
            return self._get_task_step_response(
                AgentChatResponse(response=chat_response, sources=[]),
                step,
                None,
                None,
                is_done=True,
            )
        
        observation, search_steps = self._extract_exploring_step(chat_response)
        # push back search steps to the queue
        for search_step in search_steps:
            # if search_step in history, skip
            if search_step in task.extra_state["action_history"]:
                continue
            task.extra_state["search_queue"].put(search_step)
            task.extra_state["action_history"].append(search_step)
        # print current queue size
        logger.info(f"Current search queue size: {task.extra_state['search_queue'].qsize()}")
        # only process the head of the queue
        head_search_step = task.extra_state["search_queue"].get()
        search_step = cast(SearchActionStep, head_search_step)
        search_result = self._process_search(
            task, tools, search_step
        )
        # pop the head of the queue after processing
        task.extra_state["search_queue"].task_done()

        # add search steps to task state
        task.extra_state["current_search"].append(search_result)
        agent_response = self._get_response(
            task.extra_state["current_search"], task.extra_state["sources"]
        )
        
        is_complete = task.extra_state["search_queue"].empty()
        # add observation feedback to new memory
        task.extra_state["new_memory"].put(ChatMessage(content=observation, role=MessageRole.ASSISTANT))

        # alternatively run search and observation steps
        task.extra_state["next_step_input"] = agent_response.response
        task.extra_state["next_step"] = self._assign_next_step(is_complete)

        return self._get_task_step_response(agent_response, step, task.extra_state["next_step"], task.extra_state["next_step_input"], False)
    

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

        functions = self._search_manager.get_search_functions()
        for function in functions:
            tool = FunctionTool.from_defaults(function)
            tools.append(tool)

        return tools

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"agent_worker": self.agent_worker}