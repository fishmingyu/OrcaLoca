"""
A search agent. Process raw response into json format.
"""

import json
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import BaseAgentWorker, Task, TaskStep, TaskStepOutput
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.llms.openai import OpenAI

from .environment.utils import get_logger
from .formatter import SearchChatFormatter, TokenCount, TokenCounter
from .output_parser import SearchOutputParser
from .search import SearchManager
from .types import SearchActionStep, SearchInput, SearchResult

logger = get_logger("search_agent")
dispatcher = get_dispatcher(__name__)


def parse_search_input_step(input: SearchInput, task: Task) -> None:
    extract_output = input.extract_output
    suspicous_code_from_tracer = extract_output.suspicous_code_from_tracer
    # for every codeinfo in suspicous_code_from_tracer, parse it into a action step

    if len(suspicous_code_from_tracer) > 0:
        for code_info in suspicous_code_from_tracer:
            query = code_info.keyword
            file_path = code_info.file_path
            search_step = SearchActionStep(
                action="search_callable",
                action_input={"query": query, "file_path": file_path},
            )
            task.extra_state["search_queue"].append(search_step)
            task.extra_state["action_history"].append(search_step)


def add_user_step_to_memory(
    step: TaskStep,
    search_input: SearchInput,
    task: Task,
) -> None:
    """Add user step to memory."""
    if "is_first" in step.step_state and step.step_state["is_first"]:
        memory = task.extra_state["new_memory"]
        parse_search_input_step(search_input, task)
        # add to new memory
        # logger.info("step input: \n" + step.input)
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        step.step_state["is_first"] = False
    else:
        # logger.info("step input: \n" + step.input)
        memory = task.extra_state["instruct_memory"]
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        # logger.info(f"Add user input to memory: {step.input}")


class SearchWorker(BaseAgentWorker):
    """OpenAI Agent worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        search_input: SearchInput = None,
        max_iterations: int = 10,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        self._llm = llm
        self._search_input = search_input
        self.callback_manager = callback_manager or llm.callback_manager
        self._max_iterations = max_iterations
        self._search_formatter = search_formatter or SearchChatFormatter()
        self._output_parser = output_parser or SearchOutputParser()
        self._token_counter = TokenCounter(llm)
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
        search_input: Optional[SearchInput] = None,
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
            search_input=search_input,
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
        is_done = False
        next_step_input: str = ""
        search_queue: deque = deque()
        action_history: List[SearchActionStep] = []
        current_search: List[SearchResult] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()
        instruct_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "is_done": is_done,
            "next_step_input": next_step_input,
            "search_queue": search_queue,
            "action_history": action_history,
            "current_search": current_search,
            "new_memory": new_memory,
            "instruct_memory": instruct_memory,
            "token_cnts": list(),
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
    ) -> Tuple[str, bool, List[SearchActionStep]]:
        """Extract search step."""
        # parse the output
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        try:
            obseravtion, relevance, explore_step = self._output_parser.parse_explore(
                message_content
            )
        except Exception as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        return obseravtion, relevance, explore_step

    def _bug_location_calibrate(self, output_str: str) -> str:
        """Calibrate bug location."""
        data = self._output_parser.parse_bug_report(output_str)
        for bug in data["bug_locations"]:
            file_path = bug["file"]
            # check each "file" in bug_location whether is a valid file path
            # for example the correct file should be like "astropy/io/fits/fitsrec.py",
            # the wrong file would be "/astropy__astropy/astropy/io/fits/fitsrec.py"
            # if the file is wrong, we should remove the first "/" and the first word before the first "/"
            # if the file is correct, we should keep it
            file_path = bug["file"]
            if file_path[0] == "/":
                file_path = file_path[1:]
                file_path = file_path[file_path.find("/") + 1 :]
                bug["file"] = file_path
        # logger.info(f"Bug location: {data}")
        return json.dumps(data)

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
                event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        else:
            tool_output = ToolOutput(
                content=f"Error: Tool {search_step.action} not found.",
                tool_name=search_step.action,
                raw_input={"kwargs": search_step.action_input},
                raw_output=None,
                is_error=True,
            )
        search_result = SearchResult(
            search_action=search_step.action,
            search_action_input=search_step.action_input,
            search_content=tool_output.content,
        )

        return search_result

    def _concat_search_results(self, search_results: List[SearchResult]) -> str:
        """Join and Concatenate search results."""
        search_results_str = ""
        for search_result in search_results:
            search_results_str += search_result.get_content() + "\n"
        return search_results_str

    def _check_action_valid(
        self, action: SearchActionStep, action_history: List[SearchActionStep]
    ) -> bool:
        """Check if the action is valid."""
        # first check if the action is in the history
        if action in action_history:
            return False
        # if we search_query is in the history, we don't need to call search_callable or search_class or search_func
        search_query = ""
        if action.action == "search_class_skeleton":
            search_query = action.action_input["class_name"]
        for history_action in action_history:
            if history_action.action == "search_callable":
                if search_query == history_action.action_input["query"]:
                    return False

        return True

    def _del_previous_inst_input(self, memory: ChatMemoryBuffer) -> None:
        """previous user instruction in chat message will affect the future result, so we need to delete them"""
        memory.reset()

    def _get_response(
        self,
        current_res: SearchResult,
    ) -> AgentChatResponse:
        response_str = current_res.get_content()
        return AgentChatResponse(response=response_str)

    def _get_task_step_response(
        self,
        agent_response: AGENT_CHAT_RESPONSE_TYPE,
        step: TaskStep,
        next_step: str,
        next_step_input: str,
        is_done: bool = False,
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        elif next_step == "conclusion":
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input="""Now let's come to a conclusion. Please produce the bug locations.
                    Please don't generate observations or new_search_actions. \n
                    It's time for CONCLUSION! \n
                    \n""",  # this step is conclusion
                )
            ]
        elif next_step == "explore":
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input="Please provide observation feedback and new_search_actions on the search results below. \n"
                    + next_step_input,  # this step is observation
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
                step=step,
                search_input=self._search_input,
                task=task,
            )
        tools = self.get_tools(task.input)
        # add task input to chat history
        input_chat = self._search_formatter.format(
            "CONCLUSION" if task.extra_state["is_done"] else "REGULAR",
            tools,
            chat_history=task.extra_state["instruct_memory"].get_all()
            + task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            current_search=task.extra_state["current_search"],
            current_queue=task.extra_state["search_queue"],
        )
        # if task.extra_state["is_done"]:
        #     logger.info(input_chat)
        self._del_previous_inst_input(task.extra_state["instruct_memory"])
        # send prompt
        in_token_cnt = self._token_counter.count(
            self._llm.messages_to_prompt(input_chat)
        )
        if isinstance(self._llm, OpenAI):
            chat_response = self._llm.chat(
                input_chat, response_format={"type": "json_object"}
            )
        else:
            chat_response = self._llm.chat(input_chat)
        out_token_cnt = self._token_counter.count(chat_response.message.content)
        token_cnt = TokenCount(in_token_cnt=in_token_cnt, out_token_cnt=out_token_cnt)
        logger.info(token_cnt)
        task.extra_state["token_cnts"].append(("Searcher step", token_cnt))

        logger.info(f"Chat response: {chat_response}")
        if task.extra_state["is_done"]:
            # convert the chat response to str
            cali_str = self._bug_location_calibrate(chat_response.message.content)
            return self._get_task_step_response(
                AgentChatResponse(response=cali_str, sources=[]),
                step,
                None,
                None,
                is_done=True,
            )

        observation, relevance, search_steps = self._extract_exploring_step(
            chat_response
        )
        # push back search steps to the queue
        for search_step in search_steps:
            # if search_step in history, skip
            if not self._check_action_valid(
                search_step, task.extra_state["action_history"]
            ):
                continue
            task.extra_state["search_queue"].append(search_step)
            task.extra_state["action_history"].append(search_step)
        # print current queue size
        logger.info(
            f"Current search queue size: {len(task.extra_state['search_queue'])}"
        )
        is_complete = len(task.extra_state["search_queue"]) == 0
        # add observation feedback to new memory if relevance is True
        if relevance:
            task.extra_state["new_memory"].put(
                ChatMessage(content=observation, role=MessageRole.ASSISTANT)
            )
        if is_complete:
            task.extra_state["is_done"] = True
            return self._get_task_step_response(
                AgentChatResponse(response=observation, sources=[]),
                step,
                "conclusion",
                None,
                is_done=False,
            )
        # pop the head of the queue
        head_search_step = task.extra_state["search_queue"].popleft()
        search_step = cast(SearchActionStep, head_search_step)
        search_result = self._process_search(task, tools, search_step)
        # logger.info(f"Search result: {search_result}")

        agent_response = self._get_response(search_result)
        # logger.info(f"Agent response: {agent_response.response}")

        # add search result to history
        if relevance:
            # add search steps to task state
            task.extra_state["current_search"].append(search_result)

        # alternatively run search and observation steps
        task.extra_state["next_step_input"] = agent_response.response
        # logger.info(f"Search action: {search_step.action}, Search input: {search_step.action_input}")
        # logger.info(f"Searched: {search_result.get_content()}")
        # task.extra_state["next_step"] = self._assign_next_step(is_complete)

        return self._get_task_step_response(
            agent_response, step, "explore", task.extra_state["next_step_input"], False
        )

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

        token_cnts: List[Tuple[str, TokenCount]] = task.extra_state["token_cnts"]
        in_token_cnt = 0
        out_token_cnt = 0
        for tag, token_cnt in token_cnts:
            in_token_cnt += token_cnt.in_token_cnt
            out_token_cnt += token_cnt.out_token_cnt
            logger.info(
                (
                    f"{tag:<25}: "
                    f"in {token_cnt.in_token_cnt:<5} tokens, "
                    f"out {token_cnt.out_token_cnt:<5} tokens"
                )
            )
        logger.info(
            (
                f"{'Total cnt':<25}: "
                f"in {in_token_cnt:<5} tokens, "
                f"out {out_token_cnt:<5} tokens"
            )
        )

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
        search_input: SearchInput = None,
        repo_path: str = "",
        max_iterations: int = 20,
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
            search_input=search_input,
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
