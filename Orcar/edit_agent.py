"""
A search agent. Process raw response into json format.
"""

import json
import uuid
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

from .editor import Editor
from .formatter import EditChatFormatter, TokenCount, TokenCounter
from .log_utils import get_logger
from .output_parser import EditOutputParser
from .types import EditActionStep, EditInput, EditOutput

logger = get_logger(__name__)
dispatcher = get_dispatcher(__name__)


class EditWorker(BaseAgentWorker):
    """Edit Agent worker."""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        edit_input: EditInput = None,
        max_iterations: int = 10,
        edit_manager: Editor = None,
        edit_formatter: Optional[EditChatFormatter] = None,
        output_parser: Optional[EditOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        self._llm = llm
        self._edit_input = edit_input
        self.callback_manager = callback_manager or llm.callback_manager
        self._max_iterations = max_iterations
        self._edit_manager = edit_manager
        self._edit_formatter = edit_formatter or EditChatFormatter()
        self._output_parser = output_parser or EditOutputParser()
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
        edit_input: Optional[EditInput] = None,
        max_iterations: int = 10,
        edit_manager: Editor = None,
        edit_formatter: Optional[EditChatFormatter] = None,
        output_parser: Optional[EditOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "EditWorker":
        """Convenience constructor method from set of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            EditWorker
        """
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            edit_input=edit_input,
            max_iterations=max_iterations,
            edit_manager=edit_manager,
            edit_formatter=edit_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        sys_header = self._edit_formatter.system_header
        return {"system_prompt": PromptTemplate(sys_header)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "system_prompt" in prompts:
            sys_prompt = cast(PromptTemplate, prompts["system_prompt"])
            self._edit_formatter.system_header = sys_prompt.template

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        is_done = False
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # initialize task state
        task_state = {
            "is_done": is_done,
            "new_memory": new_memory,
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

    def _extract_editing_step(self, output: ChatResponse) -> EditOutput:
        """Extract search step."""
        # parse the output
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        try:
            edit_output = self._output_parser.parse(message_content)
            # logger.info("potential_bugs: " + str(potential_bugs))
        except Exception as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        return edit_output

    def _process_edit_input(self, input: EditInput) -> Tuple[str, str]:
        """Process edit input."""
        bug_locations = input.bug_locations
        # concat bug locations into a json like this
        """
        "bug_info": [
            {
                "file_name": "path/to/file",
                "func_name": "function_name",
                "range": "line numbers",
            },
            {
                "file_name": "path/to/file",
                "func_name": "function_name",
                "range": "line numbers",
            },
        ]
        """
        bug_info = []
        ref_info = []
        for bug in bug_locations:
            revise_info = self._edit_manager._get_bug_code(
                bug.method_name, bug.file_name
            )
            if bug.class_name != "":
                reference_class_name = bug.class_name
                reference_class_info = self._edit_manager._get_bug_code(
                    reference_class_name, bug.file_name
                )
                start_line = reference_class_info.start_line
                end_line = reference_class_info.end_line
                range_str = f"{start_line}-{end_line}"
                ref_info.append(
                    {
                        "file_name": reference_class_info.file,
                        "class_name": bug.class_name,
                        "range": range_str,
                    }
                )

            start_line = revise_info.start_line
            end_line = revise_info.end_line
            range_str = f"{start_line}-{end_line}"
            bug_info.append(
                {
                    "file_name": revise_info.file,
                    "func_name": bug.method_name,
                    "range": range_str,
                }
            )
        bug_info_json = json.dumps({"bug_info": bug_info})
        ref_info_json = json.dumps({"references": ref_info})
        return bug_info_json, ref_info_json

    def _process_editor_tool(
        self,
        task: Task,
        tools: Sequence[BaseTool],
        edit_step: EditActionStep,
    ) -> str:
        tools_dict: Dict[str, BaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        # try to call the tools
        tool = tools_dict["editor_execute"]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: edit_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            try:
                dispatcher.event(
                    AgentToolCallEvent(
                        arguments=json.dumps({**edit_step.action_input}),
                        tool=tool.metadata,
                    )
                )
                tool_output = tool.call(**edit_step.action_input)

            except Exception as e:
                tool_output = ToolOutput(
                    content=f"Error: {e!s}",
                    tool_name=tool.metadata.name,
                    raw_input={"kwargs": edit_step.action_input},
                    raw_output=e,
                    is_error=True,
                )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        return tool_output.content

    def _get_response(
        self,
        current_res: EditOutput,
        tool_output: str | None = None,
    ) -> AgentChatResponse:
        # concat list tool_output and current_res.get_content
        if tool_output is None:
            response_str = current_res.get_content()
        else:
            response_str = f"{current_res.get_content()}\n Tool output:{tool_output}"

        return AgentChatResponse(response=response_str)

    def _get_is_done(self, edit_output: EditOutput) -> bool:
        """Get is done."""
        # check PATCH COMPLETE in feedback
        return "PATCH COMPLETE" in edit_output.get_content()

    def _get_task_step_response(
        self,
        agent_response: AGENT_CHAT_RESPONSE_TYPE,
        step: TaskStep,
        is_done: bool = False,
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def add_user_step_to_memory(
        self,
        step: TaskStep,
        task: Task,
    ) -> None:
        """Add user step to memory."""
        if "is_first" in step.step_state and step.step_state["is_first"]:
            memory = task.extra_state["new_memory"]
            # logger.info("step input: \n" + step.input)
            # the first the problem statement
            problem_statement = (
                f"""<Problem Statement>{step.input}</Problem Statement>"""
            )
            memory.put(ChatMessage(content=problem_statement, role=MessageRole.USER))
            step.step_state["is_first"] = False

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        # TODO: see if we want to do step-based inputs
        if step.input is not None:
            self.add_user_step_to_memory(
                step=step,
                task=task,
            )

        tools = self.get_tools(task.input)
        bug_code_input, ref_code_input = self._process_edit_input(self._edit_input)
        # add task input to chat history
        input_chat = self._edit_formatter.format(
            tools=tools,
            chat_history=task.extra_state["new_memory"].get_all(),
            bug_code_input=bug_code_input,
            reference_code=ref_code_input,
        )
        logger.info(f"Input chat: {input_chat}")

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
        logger.info(f"Chat response: {chat_response}")
        edit_output = self._extract_editing_step(chat_response)
        edit_action_step = EditActionStep(
            action_input=edit_output.action_input,
        )
        is_done = self._get_is_done(edit_output)
        task.extra_state["is_done"] = is_done

        if not is_done:
            tool_feedback = self._process_editor_tool(
                task=task,
                tools=tools,
                edit_step=edit_action_step,
            )
            logger.info(f"Tool feedback: {tool_feedback}")
            agent_response = self._get_response(edit_output, tool_feedback)
        else:
            patch = self._edit_manager.create_patch()
            agent_response = AgentChatResponse(response=patch)
        # add agent response to chat history
        task.extra_state["new_memory"].put(
            ChatMessage(content=agent_response.response, role=MessageRole.USER)
        )
        if is_done:
            task.extra_state["last_response"] = agent_response.response

        return self._get_task_step_response(agent_response, step, is_done=is_done)

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


class EditAgent(AgentRunner):
    """Edit Agent"""

    def __init__(
        self,
        llm: LLM,
        edit_input: EditInput = None,
        repo_path: str = "",
        max_iterations: int = 20,
        edit_formatter: Optional[EditChatFormatter] = None,
        output_parser: Optional[EditOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager

        self._edit_manager = Editor(repo_path=repo_path)
        self._tools = self._setup_tools()

        step_engine = EditWorker(
            tools=self._tools,
            llm=llm,
            edit_input=edit_input,
            max_iterations=max_iterations,
            edit_manager=self._edit_manager,
            edit_formatter=edit_formatter,
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

        function = self._edit_manager.editor_execute
        tool = FunctionTool.from_defaults(function)
        tools.append(tool)

        return tools

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"agent_worker": self.agent_worker}
