"""
A search agent. Process raw response into json format.
"""

import configparser
import json
import os
import traceback
import uuid
from queue import PriorityQueue
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

from .code_scorer import CodeScorer
from .formatter import SearchChatFormatter, TokenCount, TokenCounter
from .log_utils import get_logger
from .output_parser import SearchOutputParser
from .search import SearchManager
from .types import (
    BugLocations,
    HeuristicSearchResult,
    SearchActionHistory,
    SearchActionStep,
    SearchInput,
    SearchQueue,
    SearchResult,
)
from .utils import check_observation_similarity

logger = get_logger(__name__)
dispatcher = get_dispatcher(__name__)
logger_action_history = get_logger("action_history")
logger_queue = get_logger("search_queue")


def parse_search_input_step(input: SearchInput, task: Task) -> None:
    trace_analysis_output = input.trace_analysis_output
    suspicious_code_from_tracer = trace_analysis_output.suspicious_code_from_tracer
    # for every codeinfo in suspicious_code_from_tracer, parse it into a action step

    if len(suspicious_code_from_tracer) > 0:
        for code_info in suspicious_code_from_tracer:
            query = code_info.keyword
            file_path = code_info.file_path
            containing_class = code_info.class_name
            if containing_class == "":  # None
                search_step = SearchActionStep(
                    search_action="search_callable",
                    search_action_input={"query_name": query, "file_path": file_path},
                )
            else:
                search_step = SearchActionStep(
                    search_action="search_method_in_class",
                    search_action_input={
                        "class_name": containing_class,
                        "method_name": query,
                        "file_path": file_path,
                    },
                )
            task.extra_state["search_queue"].append(search_step)
            task.extra_state["action_history"].add_action(search_step)


def add_user_step_to_memory(
    step: TaskStep,
    search_input: SearchInput,
    task: Task,
) -> bool:
    """Add user step to memory."""
    if "is_first" in step.step_state and step.step_state["is_first"]:
        memory = task.extra_state["new_memory"]
        parse_search_input_step(search_input, task)
        # add to new memory
        # logger.info("step input: \n" + step.input)
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        step.step_state["is_first"] = False
        return True
    else:
        # logger.info("step input: \n" + step.input)
        memory = task.extra_state["instruct_memory"]
        memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
        # logger.info(f"Add user input to memory: {step.input}")
        return False


class SearchWorker(BaseAgentWorker):
    """Search Agent worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        search_input: SearchInput = None,
        max_iterations: int = 10,
        search_manager: SearchManager = None,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        config_path: str = "search.cfg",
    ) -> None:
        self._llm = llm
        self._search_input = search_input
        self._problem_statement = search_input.problem_statement
        self.callback_manager = callback_manager or llm.callback_manager
        self._max_iterations = max_iterations

        # Load configuration from INI file or use defaults
        self._config_dict = load_config_from_ini(config_path)

        self._search_manager = search_manager
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
        search_manager: SearchManager = None,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        config_path: str = "search.cfg",
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
            search_manager=search_manager,
            search_formatter=search_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            config_path=config_path,
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
        last_observation: str = ""
        search_queue: SearchQueue = SearchQueue(self._config_dict["priority_dict"])
        action_history: SearchActionHistory = SearchActionHistory()
        current_search: List[SearchResult] = []
        searched_node_set: set = set()
        search_cache: List[SearchResult] = []
        similarity_cache: List[bool] = []
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
            "searched_node_set": searched_node_set,
            "search_cache": search_cache,
            "similarity_cache": similarity_cache,
            "new_memory": new_memory,
            "last_observation": last_observation,
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
    ) -> Tuple[str, List[BugLocations], List[SearchActionStep]]:
        """Extract search step."""
        # parse the output
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        try:
            obseravtion, potential_bugs, explore_step = (
                self._output_parser.parse_explore(message_content)
            )
            # logger.info("potential_bugs: " + str(potential_bugs))
        except Exception as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        return obseravtion, potential_bugs, explore_step

    def _decode_bug_location(
        self, search_result: SearchResult
    ) -> Optional[Dict[str, str]]:
        """Decode bug location based on search action.

        Returns:
            Dict containing file_path, class_name, and method_name if valid, None otherwise.
        """
        search_input = search_result.search_action_input
        if not search_input:
            return None

        if search_result.search_action == "search_callable":
            query_name = search_input.get("query_name", "")
            s_q = search_result.get_search_input()
            # use get_frame_from_history to get the frame of the search result
            frame = self._search_manager.get_frame_from_history(
                search_result.search_action, s_q
            )
            file_path = frame["file_path"]
            # logger.info(f"frame: {frame}, search_input: {search_input}, search_result: {search_result}")
            query_type = frame["query_type"]
            if query_type == "method" or query_type == "function":
                return {
                    "file_path": file_path,
                    "class_name": "",  # search_callable doesn't have class name
                    "method_name": query_name,
                }
            elif query_type == "class":
                return {
                    "file_path": file_path,
                    "class_name": query_name,
                    "method_name": "",
                }
        elif search_result.search_action == "search_method_in_class":
            s_q = search_result.get_search_input()
            frame = self._search_manager.get_frame_from_history(
                search_result.search_action, s_q
            )
            file_path = frame["file_path"]
            class_name = search_input.get("class_name", "")
            method_name = search_input.get("method_name", "")
            if file_path and class_name:
                return {
                    "file_path": file_path,
                    "class_name": class_name,
                    "method_name": method_name,
                }
        elif search_result.search_action == "search_class":
            s_q = search_result.get_search_input()
            frame = self._search_manager.get_frame_from_history(
                search_result.search_action, s_q
            )
            file_path = frame["file_path"]
            class_name = search_input.get("class_name", "")
            if file_path and class_name:
                return {
                    "file_path": file_path,
                    "class_name": class_name,
                    "method_name": "",  # search_class doesn't have method name
                }
        elif search_result.search_action == "search_file_contents":
            file_path = search_input.get("file_name", "")
            if file_path:
                return {"file_path": file_path, "class_name": "", "method_name": ""}
        return None

    def _search_output_parser(
        self,
        output_str: str,
        last_observation: str,
        search_cache: List[SearchResult],
        top_k_retrieval_mode: bool = False,
    ) -> str:
        """Calibrate bug location.

        Args:
            output_str: The output string to parse
            last_observation: The last observation
            task: The current task
            top_k_retrieval_mode: If True, returns top k methods/functions and top k files separately
        """
        # Process search cache for retrieval mode if enabled
        if top_k_retrieval_mode:
            top_k = self._config_dict["top_k_output"]

            if len(search_cache) > 0:
                # Retrieval mode: just take top k overall
                bug_locations = []
                for result in search_cache[:top_k]:
                    bug_location = self._decode_bug_location(result)
                    # logger.debug(f"Output: bug_location: {bug_location}")
                    if bug_location:
                        bug_locations.append(bug_location)

                # Get non-duplicated file names for top_files_retrieved
                top_files_retrieved = []
                seen_files = set()
                for location in bug_locations:
                    file_path = location.get("file_path", "")
                    if file_path and file_path not in seen_files:
                        top_files_retrieved.append(file_path)
                        seen_files.add(file_path)

                search_output = {
                    "conclusion": last_observation,
                    "top_k_bug_locations": bug_locations,
                    "top_k_files_retrieved": top_files_retrieved,
                }
                logger.debug(f"top_k_bug_locations: {bug_locations}")
                return json.dumps(search_output)
            else:
                logger.warning("No search cache available.")
                return json.dumps({"conclusion": last_observation, "bug_locations": []})

        # If not in retrieval mode or retrieval mode failed, try normal parsing
        try:
            data = self._output_parser.parse_bug_report(output_str)
            for bug in data["bug_locations"]:
                file_path = bug["file_path"]
                # check each "file_path" in bug_location whether is a valid file path
                # for example the correct file should be like "astropy/io/fits/fitsrec.py",
                # the wrong file would be "/astropy__astropy/astropy/io/fits/fitsrec.py"
                # if the file is wrong, we should remove the first "/" and the first word before the first "/"
                # if the file is correct, we should keep it
                file_path = bug["file_path"]
                if file_path[0] == "/":
                    file_path = file_path[1:]
                    file_path = file_path[file_path.find("/") + 1 :]
                    bug["file_path"] = file_path
                class_name = bug["class_name"]
                method_name = bug["method_name"]
                # check method_name is a valid method name
                if method_name != "" and class_name != "":
                    bug_query = f"{file_path}::{class_name}::{method_name}"
                    exact_loc = self._search_manager._get_exact_loc(bug_query)
                    if exact_loc is None:
                        # revise method_name to "" since the method_name is not valid
                        bug["method_name"] = ""

            # cat last observation and bug location
            search_output = {
                "conclusion": last_observation,
                "bug_locations": data["bug_locations"],
            }
            return json.dumps(search_output)
        except Exception as e:
            logger.warning(
                f"Failed to parse bug report: {e}. Falling back to search cache."
            )
            # Get top K results from search_cache for output
            top_k = self._config_dict["top_k_output"]
            if len(search_cache) > 0:
                # Original mode: just take top k overall
                bug_locations = []
                for result in search_cache[:top_k]:
                    bug_location = self._decode_bug_location(result)
                    if bug_location:
                        bug_locations.append(bug_location)

                search_output = {
                    "conclusion": last_observation,
                    "bug_locations": bug_locations,
                }
                return json.dumps(search_output)
            else:
                # If no search cache available, return empty result
                search_output = {"conclusion": last_observation, "bug_locations": []}
                return json.dumps(search_output)

    def _process_search_queue(
        self,
        task: Task,
        tools: Sequence[BaseTool],
    ) -> List[SearchResult] | None:
        """Process search queue in batches.

        Args:
            task: The current task
            tools: Available tools

        Returns:
            List of search results or None if queue is empty
        """
        if len(task.extra_state["search_queue"]) == 0:
            return None

        results = []
        processed_count = 0

        while (
            len(task.extra_state["search_queue"]) > 0
            and processed_count < self._config_dict["batch_size"]
        ):
            if self._config_dict[
                "redundancy_control"
            ]:  # if redundancy control is enabled
                # while not duplicate, keep popping
                while len(task.extra_state["search_queue"]) > 0:
                    head_search_step = task.extra_state["search_queue"].pop()
                    search_step = cast(SearchActionStep, head_search_step)
                    search_result = self._process_search_action(tools, search_step)
                    # check if the search result is duplicate
                    is_duplicate = self._check_search_result_duplicate(search_result)
                    # if duplicate, continue to pop
                    if not is_duplicate:  # until we get a non-duplicate search result
                        results.append(search_result)
                        processed_count += 1
                        break
                    logger_queue.info(f"Duplicate search result: {search_result}")
            else:  # if redundancy control is disabled
                head_search_step = task.extra_state["search_queue"].pop()
                search_step = cast(SearchActionStep, head_search_step)
                search_result = self._process_search_action(tools, search_step)
                results.append(search_result)
                processed_count += 1

        return results if results else None

    def _process_search_action(
        self,
        tools: Sequence[BaseTool],
        search_step: SearchActionStep,
    ) -> SearchResult:
        tools_dict: Dict[str, BaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        # try to call the tools
        if search_step.search_action in tools_dict:
            tool = tools_dict[search_step.search_action]
            with self.callback_manager.event(
                CBEventType.FUNCTION_CALL,
                payload={
                    EventPayload.FUNCTION_CALL: search_step.search_action_input,
                    EventPayload.TOOL: tool.metadata,
                },
            ) as event:
                try:
                    dispatcher.event(
                        AgentToolCallEvent(
                            arguments=json.dumps({**search_step.search_action_input}),
                            tool=tool.metadata,
                        )
                    )
                    tool_output = tool.call(**search_step.search_action_input)

                except Exception as e:
                    tool_output = ToolOutput(
                        content=f"Error: {e!s}",
                        tool_name=tool.metadata.name,
                        raw_input={"kwargs": search_step.search_action_input},
                        raw_output=e,
                        is_error=True,
                    )
                event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        else:
            tool_output = ToolOutput(
                content=f"Error: Tool {search_step.search_action} not found.",
                tool_name=search_step.search_action,
                raw_input={"kwargs": search_step.search_action_input},
                raw_output=None,
                is_error=True,
            )
        search_result = SearchResult(
            search_action=search_step.search_action,
            search_action_input=search_step.search_action_input,
            search_content=tool_output.content,
        )

        return search_result

    def _check_search_result_skeleton(self, search_result: SearchResult) -> bool:
        """Check if the search result is a Class Skeleton:"""
        action = search_result.search_action
        search_input = search_result.get_search_input()
        # use get_frame_from_history to get the frame of the search result
        frame = self._search_manager.get_frame_from_history(action, search_input)
        # check frame's is_skeleton is True or False
        if frame is not None:
            is_skeleton = frame["is_skeleton"]
            if is_skeleton:
                return True
        return False

    def _check_search_result_duplicate(self, search_result: SearchResult) -> bool:
        """Check if search query of the search result is duplicate."""
        action = search_result.search_action
        search_input = search_result.get_search_input()
        # use get_query_from_history to get the query of the search result
        query = self._search_manager.get_query_from_history(action, search_input)
        # check query is not None
        if query is not None:
            # use search_manager.check_and_add_query to check if the query is duplicate
            is_duplicate = self._search_manager.check_and_add_query(query)
            return is_duplicate
        return False

    def _check_search_result_type(self, search_result: SearchResult, type: str) -> bool:
        """Check if the search result is a certain type."""
        action = search_result.search_action
        search_input = search_result.get_search_input()
        # use get_frame_from_history to get the frame of the search result
        frame = self._search_manager.get_frame_from_history(action, search_input)
        # check frame is not None
        if frame is not None:
            query_type = frame["query_type"]
            if query_type == type:
                return True
        return False

    # enlarge search space
    def _get_search_result_file_path(self, search_result: SearchResult) -> str | None:
        """Get the file path of the search result."""
        action = search_result.search_action
        search_input = search_result.get_search_input()
        frame = self._search_manager.get_frame_from_history(action, search_input)
        if frame is not None:
            file_path = frame["file_path"]
            # try to use search_manager to check the node existence
            exist = self._search_manager.get_node_existence(file_path)
            if exist:
                return file_path
        return None

    def _concat_search_results(self, search_results: List[SearchResult]) -> str:
        """Join and Concatenate search results."""
        search_results_str = ""
        for search_result in search_results:
            search_results_str += search_result.get_content() + "\n"
        return search_results_str

    def _search_result_heuristic(
        self, search_result: SearchResult, potential_bugs: List[BugLocations]
    ) -> HeuristicSearchResult:
        """Heuristic to determine if the search result is a bug location."""
        search_action = search_result.search_action
        search_input = search_result.get_search_input()

        search_query = self._search_manager.get_query_from_history(
            action=search_action, input=search_input
        )
        is_disambiguation = self._check_search_result_type(
            search_result, "disambiguation"
        )
        is_file = self._check_search_result_type(search_result, "file")
        is_skeleton = self._check_search_result_skeleton(search_result)
        if (
            is_disambiguation or is_skeleton or is_file
        ):  # not valid if is disambiguation, don't need to check the node existance
            # not valid if is skeleton, don't need to check the node existance
            # not valid if is file(file info << method info), don't need to check the node existance
            # Reason: disambiguation and skeleton are limited information, and we have already
            # use the action decomposition to get the detailed information
            valid_search = False
        else:
            # valid depends on the node existence
            valid_search = self._search_manager.get_node_existence(search_query)

        if valid_search is False:
            return HeuristicSearchResult(
                heuristic=-1, search_result=search_result
            )  # -1 means drop this search result

        weighted_heuristic = 0
        valid_bug_count = 0
        for bug in potential_bugs:
            bug_query = bug.bug_query()
            vaild_bug = (
                bug_query is not None
            ) and self._search_manager.get_node_existence(bug_query)
            if vaild_bug:
                heuristic = self._search_manager.get_distance_between_queries(
                    search_query, bug_query
                )
                weighted_heuristic += heuristic
                valid_bug_count += 1
            else:
                continue  # if the bug_query is not valid, we should continue to check the next bug
        if valid_bug_count == 0:
            return HeuristicSearchResult(
                heuristic=0, search_result=search_result
            )  # normal search result
        else:
            weighted_heuristic = weighted_heuristic / valid_bug_count
        return HeuristicSearchResult(
            heuristic=weighted_heuristic, search_result=search_result
        )

    def _check_action_valid(
        self, action: SearchActionStep, action_history: SearchActionHistory
    ) -> bool:
        """Check if the action is valid."""
        # first check if the action is in the history
        if action_history.check_action(action):
            return False  # non-valid

        # special case for search_class; if search_callable with the same query is in the history, skip
        if action.search_action == "search_class":
            # case 1: search_class argument file_path is None
            if "file_path" not in action.search_action_input:
                for history_action in action_history.keys():
                    if history_action.search_action == "search_callable":
                        if (
                            history_action.search_action_input["query_name"]
                            == action.search_action_input[
                                "class_name"
                            ]  # action here is search_class
                            and "file_path" not in history_action.search_action_input
                        ):
                            return False
            # case 2: search_class argument file_path is not None
            else:
                for history_action in action_history.keys():
                    # if "file_path" not in history_action.search_action_input, skip
                    if "file_path" not in history_action.search_action_input:
                        continue
                    if history_action.search_action == "search_callable":
                        if (
                            history_action.search_action_input["query_name"]
                            == action.search_action_input[
                                "class_name"
                            ]  # action here is search_class
                            and history_action.search_action_input["file_path"]
                            == action.search_action_input["file_path"]
                        ):  # hit
                            return False
        # special case for search_callable; if search_class with the same query is in the history, skip
        if action.search_action == "search_callable":
            # case 1: search_callable argument file_path is None
            if "file_path" not in action.search_action_input:
                for history_action in action_history.keys():
                    if history_action.search_action == "search_class":
                        if (
                            history_action.search_action_input["class_name"]
                            == action.search_action_input[
                                "query_name"
                            ]  # action here is search_callable
                            and "file_path" not in history_action.search_action_input
                        ):
                            return False
            # case 2: search_callable argument file_path is not None
            else:
                for history_action in action_history.keys():
                    # if "file_path" not in history_action.search_action_input, skip
                    if "file_path" not in history_action.search_action_input:
                        continue
                    if history_action.search_action == "search_class":
                        if (
                            history_action.search_action_input["class_name"]
                            == action.search_action_input[
                                "query_name"
                            ]  # action here is search_callable
                            and history_action.search_action_input["file_path"]
                            == action.search_action_input["file_path"]
                        ):
                            return False
        return True

    def _class_methods_ranking(
        self,
        search_result: SearchResult,
        task: Task,
    ) -> List[SearchActionStep]:
        """Ranking the class methods."""
        # if the action is search_class, we should rank the class methods
        search_action = search_result.search_action

        is_class = self._check_search_result_type(search_result, "class")
        if is_class:
            frame = self._search_manager.get_frame_from_history(
                search_action, search_result.get_search_input()
            )
            search_query = frame["search_query"]
            file_path = frame["file_path"]
            # search_query is the exact node name
            class_methods, methods_code = self._search_manager._get_class_methods(
                search_query
            )
            if len(class_methods) == 0:
                return []
            # package the list of methods into a list of ChatMessage
            chat_messages: List[List[ChatMessage]] = []
            for method in methods_code:
                chat_messages.append(
                    [ChatMessage(role=MessageRole.USER, content=method)]
                )
            logger.info(f"Class methods number: {len(class_methods)}")
            code_scorer = CodeScorer(
                llm=self._llm, problem_statement=self._problem_statement
            )
            # score the list of methods
            scores = code_scorer.score_batch(chat_messages)
            task.extra_state["token_cnts"].append(
                ("Methods Score", code_scorer.get_sum_cnt())
            )
            # combine the scores with the method names
            results = []
            for i, method in enumerate(class_methods):
                results.append({"method_name": method, "score": scores[i]})
            sorted_results = sorted(
                results, key=lambda x: x["score"], reverse=True
            )  # from high to low
            # prune scores less than self._config_dict["score_threshold"]
            sorted_results = [
                result
                for result in sorted_results
                if result["score"] > self._config_dict["score_threshold"]
            ]
            # get top 2 methods
            top_k = self._config_dict["top_k_methods"]
            if len(sorted_results) < top_k:
                top_k = len(sorted_results)
            search_steps = []
            for i in range(top_k):
                method_name = sorted_results[i]["method_name"].split("::")[-1]
                class_name = sorted_results[i]["method_name"].split("::")[-2]
                search_steps.append(
                    SearchActionStep(
                        search_action="search_method_in_class",
                        search_action_input={
                            "class_name": class_name,
                            "method_name": method_name,
                            "file_path": file_path,
                        },
                    )
                )
            return search_steps
        return []

    def _file_functions_ranking(
        self,
        search_result: SearchResult,
        task: Task,
    ) -> List[SearchActionStep]:
        """Ranking the file functions."""
        # if the action is search_file_contents, we should rank the file functions
        search_action = search_result.search_action
        is_file = self._check_search_result_type(search_result, "file")
        if is_file:
            frame = self._search_manager.get_frame_from_history(
                search_action, search_result.get_search_input()
            )
            file_node_name = frame["search_query"]
            # search_query is the exact node name
            functions, functions_code = self._search_manager._get_file_functions(
                file_node_name
            )
            if len(functions) == 0:
                return []
            # package the list of functions into a list of ChatMessage
            chat_messages: List[List[ChatMessage]] = []
            for function in functions_code:
                chat_messages.append(
                    [ChatMessage(role=MessageRole.USER, content=function)]
                )
            logger.info(f"File functions number: {len(functions)}")
            code_scorer = CodeScorer(
                llm=self._llm, problem_statement=self._problem_statement
            )
            # score the list of functions
            scores = code_scorer.score_batch(chat_messages)
            task.extra_state["token_cnts"].append(
                ("Functions Score", code_scorer.get_sum_cnt())
            )
            # combine the scores with the function names
            results = []
            for i, function in enumerate(functions):
                results.append({"function_name": function, "score": scores[i]})
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            # prune scores less than self._config_dict["score_threshold"]
            sorted_results = [
                result
                for result in sorted_results
                if result["score"] > self._config_dict["score_threshold"]
            ]
            # get top 2 functions
            top_k = self._config_dict["top_k_functions"]
            if len(sorted_results) < top_k:
                top_k = len(sorted_results)
            search_steps = []
            for i in range(top_k):
                function_name = sorted_results[i]["function_name"].split("::")[-1]
                file_path = sorted_results[i]["function_name"].split("::")[0]
                search_steps.append(
                    SearchActionStep(
                        search_action="search_callable",
                        search_action_input={
                            "query_name": function_name,
                            "file_path": file_path,
                        },
                    )
                )
            return search_steps
        return []

    def _disambiguation_ranking(
        self,
        search_result: SearchResult,
        task: Task,
    ) -> List[SearchActionStep]:
        """Ranking the disambiguation."""
        # if the action is disambiguation, we should rank the disambiguation
        search_action = search_result.search_action
        search_action_input = search_result.search_action_input

        is_disambiguation = self._check_search_result_type(
            search_result, "disambiguation"
        )

        def check_action_is_class(search_action: str) -> bool:
            """Check if the action is search_class."""
            if search_action == "search_class":
                return True
            return False

        def check_action_is_file(search_action: str) -> bool:
            """Check if the action is search_file_contents."""
            if search_action == "search_file_contents":
                return True
            return False

        if is_disambiguation:
            # if is_class, we don't score
            is_class = check_action_is_class(search_action)
            if is_class:
                class_name = search_action_input["class_name"]
                file_paths = self._search_manager._get_disambiguous_classes(class_name)
                if len(file_paths) == 0:
                    return []
                search_steps = []
                for file_path in file_paths:
                    search_steps.append(
                        SearchActionStep(
                            search_action="search_class",
                            search_action_input={
                                "class_name": class_name,
                                "file_path": file_path,
                            },
                        )
                    )
                return search_steps
            # if is_file, we don't score
            is_file = check_action_is_file(search_action)
            if is_file:
                file_name = search_action_input["file_name"]
                # we don't have directory_path since we got the disambiguation
                file_paths = self._search_manager._get_disambiguous_files(file_name)
                if len(file_paths) == 0:
                    return []
                search_steps = []
                for file_path in file_paths:
                    # use relative path to get the directory_path
                    directory_path = os.path.dirname(file_path)
                    search_steps.append(
                        SearchActionStep(
                            search_action="search_file_contents",
                            search_action_input={
                                "file_name": file_name,
                                "directory_path": directory_path,
                            },
                        )
                    )
                return search_steps
            # score the methods
            # three cases:
            # 1. search_method_in_class, and no file_path in search_action_input (if file_path, then no disambiguation)
            # 2. search_callable, and no file_path in search_action_input (it may contain class, but we can deal with it: note here the only trouble is class without skeleton)
            # 3. search_callable, and file_path in search_action_input
            if search_action == "search_method_in_class":
                class_name = search_action_input["class_name"]
                method_name = search_action_input["method_name"]
                disambiguated_methods, disambiguated_methods_code = (
                    self._search_manager._get_disambiguous_methods(
                        method_name=method_name, class_name=class_name
                    )
                )
            if search_action == "search_callable":
                query_name = search_action_input["query_name"]
                if "file_path" not in search_action_input:
                    disambiguated_methods, disambiguated_methods_code = (
                        self._search_manager._get_disambiguous_methods(
                            method_name=query_name, class_name=None, file_path=None
                        )
                    )
                else:
                    file_path = search_action_input["file_path"]
                    disambiguated_methods, disambiguated_methods_code = (
                        self._search_manager._get_disambiguous_methods(
                            method_name=query_name, class_name=None, file_path=file_path
                        )
                    )

            if len(disambiguated_methods) == 0:
                return []

            # package the list of disambiguation into a list of ChatMessage
            chat_messages: List[List[ChatMessage]] = []
            for dis in disambiguated_methods_code:
                chat_messages.append([ChatMessage(role=MessageRole.USER, content=dis)])
            logger.info(f"Disambiguation number: {len(disambiguated_methods)}")
            code_scorer = CodeScorer(
                llm=self._llm, problem_statement=self._problem_statement
            )
            # score the list of disambiguation
            scores = code_scorer.score_batch(chat_messages)
            task.extra_state["token_cnts"].append(
                ("Disambiguation Score", code_scorer.get_sum_cnt())
            )
            # combine the scores with the disambiguation names
            results = []
            for i, dis in enumerate(disambiguated_methods):
                results.append({"disambiguated_method": dis, "score": scores[i]})
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            # prune scores less than self._config_dict["score_threshold"]
            sorted_results = [
                result
                for result in sorted_results
                if result["score"] > self._config_dict["score_threshold"]
            ]
            # get top 3 disambiguation
            top_k = self._config_dict["top_k_disambiguation"]
            if len(sorted_results) <= top_k:
                top_k = 1  # only one disambiguation
            if len(sorted_results) == 0:
                return []
            search_steps = []
            # please note, the disambiguated_method is the node name
            # if two "::" in the node name, it means it is a class's method
            # if one "::" in the node name, it means it is a function (why class is impossible? because we have already disambiguated the class)
            for i in range(top_k):
                disambiguated_method = sorted_results[i]["disambiguated_method"]
                # count the number of "::" in the node name
                if disambiguated_method.count("::") == 2:
                    # file_path::class_name::method_name
                    method_name = disambiguated_method.split("::")[-1]
                    class_name = disambiguated_method.split("::")[-2]
                    file_path = disambiguated_method.split("::")[0]
                    search_steps.append(
                        SearchActionStep(
                            search_action="search_method_in_class",
                            search_action_input={
                                "class_name": class_name,
                                "method_name": method_name,
                                "file_path": file_path,
                            },
                        )
                    )
                else:
                    function_name = disambiguated_method.split("::")[-1]
                    file_path = disambiguated_method.split("::")[0]
                    search_steps.append(
                        SearchActionStep(
                            search_action="search_callable",
                            search_action_input={
                                "query_name": function_name,
                                "file_path": file_path,
                            },
                        )
                    )
            return search_steps
        return []

    def _early_stop(self, task: Task) -> bool:
        """Early stop."""
        observation_history = task.extra_state["new_memory"].get_all()
        # get last 2 observations
        if len(observation_history) < 2:
            return False
        last_observation = observation_history[-1].content
        second_last_observation = observation_history[-2].content
        # check similarity
        _, is_similar = check_observation_similarity(
            last_observation,
            second_last_observation,
            self._config_dict["similarity_threshold"],
        )
        # add is_similar to the similarity_cache
        task.extra_state["similarity_cache"].append(is_similar)
        # use sliding window to check the similarity, window size is self._config_dict["sliding_window_size"]
        if (
            len(task.extra_state["similarity_cache"])
            > self._config_dict["sliding_window_size"]
        ):
            task.extra_state["similarity_cache"].pop(0)
        # if all the observations in the window are similar and size is 10, we should early stop
        if (
            len(task.extra_state["similarity_cache"])
            == self._config_dict["sliding_window_size"]
        ):
            early_stop = all(task.extra_state["similarity_cache"])
        else:
            early_stop = False
        logger.info(f"Is early stop: {early_stop} similarity: {is_similar}")
        return early_stop

    def _judge_is_complete(self, task: Task) -> bool:
        """Judge if the task is complete."""
        # first check _early_stop
        if self._early_stop(task):
            return True
        # then check if the search_queue is empty
        if len(task.extra_state["search_queue"]) == 0:
            return True
        return False

    def _process_search_result(
        self,
        search_result: SearchResult,
        task: Task,
        potential_bugs: List[BugLocations],
    ) -> AgentChatResponse:
        """Process search result."""
        # the response could contain action not found, so we should keep it
        agent_response = self._get_response(
            search_result
        )  # directly return the search result

        if search_result is not None:
            heuristic_search_result = self._search_result_heuristic(
                search_result, potential_bugs
            )
            if (
                heuristic_search_result.heuristic >= 0
            ):  # if the heuristic is greater than 0, we should add it to the current search
                # add search steps to task state
                task.extra_state["current_search"].append(
                    search_result
                )  # we tolerate duplicate nodes, since it may have different content (e.g. different actions)
        # return the original search result
        return agent_response

    def _process_heuristic_search_cache(
        self, task: Task, potential_bugs: List[BugLocations]
    ) -> None:
        # cache heuristic search results
        search_cache: PriorityQueue[HeuristicSearchResult] = PriorityQueue()
        # every step recalcuate the heuristic of the search result
        for search_result in task.extra_state["current_search"]:
            heuristic_search_result = self._search_result_heuristic(
                search_result, potential_bugs
            )
            search_query = self._search_manager.get_query_from_history(
                action=search_result.search_action,
                input=search_result.get_search_input(),
            )

            # if search_query not in searched_node_set, add it to the search_cache (no duplicate)
            if search_query not in task.extra_state["searched_node_set"]:
                search_cache.put(heuristic_search_result)
            # maintain a searched_node_set to avoid duplicate search result in search_cache
            task.extra_state["searched_node_set"].add(search_query)
        task.extra_state["searched_node_set"] = set()  # reset the searched_node_set
        # get the top k search results, put it to task.extra_state["search_cache"]
        task.extra_state["search_cache"] = []
        for _ in range(min(self._config_dict["top_k_search"], search_cache.qsize())):
            task.extra_state["search_cache"].append(search_cache.get().search_result)

    def _action_decomposition(
        self, search_result: SearchResult, task: Task, config: dict
    ) -> None:  # config is {"class": True, "disambiguation": True, "file": True}
        # we use type to determine whether the search result is a class, disambiguation or file
        # if it is a disambiguation, we should rank the disambiguation
        # if it is a class, we should rank the class methods
        # if it is a file, we should rank the file functions
        # class, disambiguation and file wouldn't appear at the same time

        # append actions to the left of the queue
        def add_decomposed_actions_to_queue(
            actions: List[SearchActionStep], log_str: str, priority: bool = False
        ) -> None:
            if len(actions) > 0:
                for action in actions:
                    if self._check_action_valid(
                        action, task.extra_state["action_history"]
                    ):
                        if priority:
                            task.extra_state["search_queue"].append_with_priority(
                                action,
                                self._config_dict["priority_dict"]["decomposition"],
                            )
                        else:
                            task.extra_state["search_queue"].append(action)
                        task.extra_state["action_history"].add_action(action)
                logger_action_history.info(f"{log_str}: {actions}")

        if config["class"]:
            top_class_actions = self._class_methods_ranking(search_result, task)
            # add top class methods to the left of the queue
            add_decomposed_actions_to_queue(
                top_class_actions, "Top class methods", priority=True
            )

        if config["file"]:
            top_function_actions = self._file_functions_ranking(search_result, task)
            # add top file functions to the left of the queue
            add_decomposed_actions_to_queue(
                top_function_actions, "Top file functions", priority=True
            )

        if config["disambiguation"]:
            disambiguation_actions = self._disambiguation_ranking(search_result, task)
            # add disambiguation to the left of the queue
            add_decomposed_actions_to_queue(
                disambiguation_actions, "Disambiguation", priority=True
            )

        # put the file content search to the left of the queue
        file_node = self._get_search_result_file_path(search_result)
        if file_node is not None:
            file_search_action = SearchActionStep(
                search_action="search_file_contents",
                search_action_input={
                    "file_name": os.path.basename(file_node),
                    "directory_path": os.path.dirname(file_node),
                },
            )
            if self._check_action_valid(
                file_search_action, task.extra_state["action_history"]
            ):
                task.extra_state["search_queue"].append_with_priority(
                    file_search_action,
                    self._config_dict["priority_dict"]["related_file"],
                )
                task.extra_state["action_history"].add_action(file_search_action)
                logger.info(f"File search: {file_node}")

    def _del_previous_inst_input(self, memory: ChatMemoryBuffer) -> None:
        """previous user instruction in chat message will affect the future result, so we need to delete them"""
        memory.reset()

    def _get_response(
        self,
        current_res: SearchResult,
    ) -> AgentChatResponse:
        response_str = current_res.get_next_response()
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

    def _get_step_str(
        self,
        is_first: bool,
        task: Task,
    ) -> str:
        if is_first:
            return "FIRST"
        elif task.extra_state["is_done"]:
            return "CONCLUSION"
        else:
            return "REGULAR"

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        # TODO: see if we want to do step-based inputs
        if step.input is not None:
            is_first = add_user_step_to_memory(
                step=step,
                search_input=self._search_input,
                task=task,
            )
        tools = self.get_tools(task.input)
        # add task input to chat history
        input_chat = self._search_formatter.format(
            self._get_step_str(is_first, task),
            tools,
            chat_history=task.extra_state["instruct_memory"].get_all()
            + task.memory.get(input=task.input)
            + task.extra_state["new_memory"].get_all(),
            current_search=task.extra_state["search_cache"],
            current_queue=task.extra_state["search_queue"],
        )
        logger.debug(f"Search content: {task.extra_state['instruct_memory'].get_all()}")
        logger.info(f"Search cache: {task.extra_state['search_cache']}")
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
        if task.extra_state["is_done"]:
            task.extra_state["token_cnts"].append(("Conclusion step", token_cnt))
        else:
            task.extra_state["token_cnts"].append(("Searcher step", token_cnt))

        logger.info(f"Chat response: {chat_response}")
        if task.extra_state["is_done"]:
            # convert the chat response to str
            search_output_str = self._search_output_parser(
                chat_response.message.content,
                task.extra_state["last_observation"],
                task.extra_state["search_cache"],
                self._config_dict["top_k_retrieval_mode"],
            )
            return self._get_task_step_response(
                AgentChatResponse(response=search_output_str, sources=[]),
                step,
                None,
                None,
                is_done=True,
            )

        observation, potential_bugs, search_steps = self._extract_exploring_step(
            chat_response
        )
        # push back search steps to the queue (LLM's recommendation)s
        for search_step in search_steps:
            # only add the search step to the queue if it is valid
            if self._check_action_valid(
                search_step, task.extra_state["action_history"]
            ):
                task.extra_state["search_queue"].append(search_step)
            task.extra_state["action_history"].add_action(
                search_step
            )  # here we would count the action if it appears multiple times (only for LLM's recommendation)
        # log action_history
        logger_action_history.debug(
            f"Action history: \n {task.extra_state['action_history']}"
        )
        # print current queue size
        logger.info(
            f"Current search queue size: {len(task.extra_state['search_queue'])}"
        )

        task.extra_state["new_memory"].put(
            ChatMessage(content=observation, role=MessageRole.ASSISTANT)
        )
        task.extra_state["last_observation"] = observation
        is_complete = self._judge_is_complete(task)
        logger.info(f"Is complete: {is_complete}")
        if is_complete:
            task.extra_state["is_done"] = True
            return self._get_task_step_response(
                AgentChatResponse(response=observation, sources=[]),
                step,
                "conclusion",
                None,
                is_done=False,
            )
        # resort the search_queue based on the action history
        # if enable priority, resort the search_queue
        enable_priority = self._config_dict["priority_dict"]["enable"]
        if enable_priority:
            task.extra_state["search_queue"].resort(task.extra_state["action_history"])
        logger_queue.debug(f"search queue: \n {task.extra_state['search_queue']}")
        # process the search queue
        search_results = self._process_search_queue(
            task, tools
        )  # this step processed search (and forms the search result)
        if search_results is None:  # return is_complete
            task.extra_state["is_done"] = True
            return self._get_task_step_response(
                AgentChatResponse(response=observation, sources=[]),
                step,
                "conclusion",
                None,
                is_done=False,
            )

        # Process each search result for action decomposition
        for search_result in search_results:
            self._action_decomposition(
                search_result, task, self._config_dict["score_decomposition"]
            )

        # Process each search result and get agent responses
        next_step_input = ""
        for search_result in search_results:
            agent_response = self._process_search_result(
                search_result, task, potential_bugs
            )
            next_step_input += agent_response.response + "\n"

        if self._config_dict["context_control"]:  # use heuristic search cache
            self._process_heuristic_search_cache(task, potential_bugs)
        else:  # directly put the search result to the search_cache
            for search_result in search_results:
                task.extra_state["search_cache"].append(search_result)

        task.extra_state["next_step_input"] = next_step_input

        return self._get_task_step_response(
            next_step_input, step, "explore", task.extra_state["next_step_input"], False
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
                    f"in {token_cnt.in_token_cnt:>6} tokens, "
                    f"out {token_cnt.out_token_cnt:>6} tokens"
                )
            )
        logger.info(
            (
                f"{'Total cnt':<25}: "
                f"in {in_token_cnt:>6} tokens, "
                f"out {out_token_cnt:>6} tokens"
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
        max_iterations: int = 10,
        search_formatter: Optional[SearchChatFormatter] = None,
        output_parser: Optional[SearchOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        config_path: str = "search.cfg",
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
            search_manager=self._search_manager,
            search_formatter=search_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            config_path=config_path,
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


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the search agent."""
    return {
        "context_control": True,
        "redundancy_control": True,
        "score_decomposition": {
            "class": True,
            "file": True,
            "disambiguation": True,
        },
        "priority_dict": {
            "enable": True,
            "basic": 1,
            "decomposition": 2,
            "related_file": 2,
        },
        "top_k_search": 12,
        "top_k_output": 3,  # Number of bug locations to include in the output
        "top_k_retrieval_mode": False,  # Whether to use retrieval mode for output
        "sliding_window_size": 15,
        "top_k_methods": 3,
        "top_k_disambiguation": 3,
        "top_k_functions": 2,
        "score_threshold": 75,
        "similarity_threshold": 0.97,
        "batch_size": 1,  # Number of actions to process in each batch
    }


def load_config_from_ini(file_path: str = "search.cfg") -> Dict[str, Any]:
    """Load configuration from an INI file.

    This function loads configuration from an INI file with the following sections:
    - [SEARCH]: General search parameters like context_control, top_k_search, etc.
    - [SCORE_DECOMPOSITION]: Controls which types of decomposition are enabled
    - [PRIORITY]: Controls priority settings for search queue management

    Args:
        file_path: Path to the INI file. If a relative path is provided, it will be
                  resolved relative to the current working directory. Default: "search.cfg"

    Returns:
        Dictionary containing the configuration with all values converted to appropriate types.
        If the file doesn't exist or can't be read, returns the default configuration.
    """
    # Start with default config
    config = get_default_config()

    # Try different locations if file_path is not absolute
    search_paths = [file_path]
    if not os.path.isabs(file_path):
        # Try in the Orcar directory
        orcar_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.append(os.path.join(orcar_dir, file_path))
        # Try in the parent directory
        search_paths.append(os.path.join(os.path.dirname(orcar_dir), file_path))

    # Try each path
    config_found = False
    for path in search_paths:
        if os.path.exists(path):
            file_path = path
            config_found = True
            break

    if not config_found:
        logger.warning(
            f"Config file not found in any of: {search_paths}. Using default configuration."
        )
        return config

    # Load from INI file
    try:
        config_parser = configparser.ConfigParser()
        config_parser.read(file_path)

        # Helper function to convert string values to their appropriate types
        def parse_value(value: str) -> Any:
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value

        # Parse SEARCH section
        if "SEARCH" in config_parser:
            for key, value in config_parser["SEARCH"].items():
                if key in config:
                    original_value = config[key]
                    parsed_value = parse_value(value)
                    config[key] = parsed_value
                    if parsed_value != original_value:
                        logger.debug(
                            f"Config: {key} changed from {original_value} to {parsed_value}"
                        )
                else:
                    logger.warning(f"Unknown config key in [SEARCH]: {key}")

        # Parse SCORE_DECOMPOSITION section
        if "SCORE_DECOMPOSITION" in config_parser:
            for key, value in config_parser["SCORE_DECOMPOSITION"].items():
                if key in config["score_decomposition"]:
                    original_value = config["score_decomposition"][key]
                    parsed_value = parse_value(value)
                    config["score_decomposition"][key] = parsed_value
                    if parsed_value != original_value:
                        logger.debug(
                            f"Config: score_decomposition.{key} changed from {original_value} to {parsed_value}"
                        )
                else:
                    logger.warning(
                        f"Unknown config key in [SCORE_DECOMPOSITION]: {key}"
                    )

        # Parse PRIORITY section
        if "PRIORITY" in config_parser:
            for key, value in config_parser["PRIORITY"].items():
                if key in config["priority_dict"]:
                    original_value = config["priority_dict"][key]
                    parsed_value = parse_value(value)
                    config["priority_dict"][key] = parsed_value
                    if parsed_value != original_value:
                        logger.debug(
                            f"Config: priority_dict.{key} changed from {original_value} to {parsed_value}"
                        )
                else:
                    logger.warning(f"Unknown config key in [PRIORITY]: {key}")

        # Check for missing sections
        required_sections = ["SEARCH", "SCORE_DECOMPOSITION", "PRIORITY"]
        for section in required_sections:
            if section not in config_parser:
                logger.warning(
                    f"Missing section [{section}] in config file. Using defaults for this section."
                )

        logger.info(f"Configuration loaded from {file_path}")
    except Exception as e:
        logger.error(f"Error loading configuration from {file_path}: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Using default configuration due to error.")

    return config
