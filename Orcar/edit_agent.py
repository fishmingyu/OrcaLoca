"""
A search agent. Process raw response into json format.
"""

import json
import uuid
from typing import Any, List, Optional, Tuple, cast

from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import BaseAgentWorker, Task, TaskStep, TaskStepOutput
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import AsyncBaseTool
from llama_index.llms.openai import OpenAI

from .editor import Editor
from .formatter import EditChatFormatter, TokenCount, TokenCounter
from .log_utils import get_logger
from .output_parser import EditOutputParser
from .types import EditBugCode, EditInput

logger = get_logger(__name__)
dispatcher = get_dispatcher(__name__)


class EditWorker(BaseAgentWorker):
    """Edit Agent worker."""

    def __init__(
        self,
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
        self._get_tools = lambda _: []

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

    def _extract_editing_step(self, output: ChatResponse) -> List[EditBugCode]:
        """Extract search step."""
        # parse the output
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        try:
            revised_code = self._output_parser.parse(message_content)
            # logger.info("potential_bugs: " + str(potential_bugs))
        except Exception as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        return revised_code

    def _process_edit_input(self, input: EditInput) -> Tuple[str, str]:
        """Process edit input."""
        problem_statement = input.problem_statement
        bug_locations = input.bug_locations
        # concat bug locations into a json like this
        """
            "bug_info": [
            {
                "file_name": "path/to/file",
                "func_name": "function_name",
                "content": "code snippet",
            },
            {
                "file_name": "path/to/file",
                "func_name": "function_name",
                "content": "code snippet",
            },
        ]
        """
        bug_info = []
        for bug in bug_locations:
            revise_info = self._edit_manager._get_bug_code(
                bug.method_name, bug.file_name
            )
            bug_info.append(
                {
                    "file_name": bug.file_name,
                    "func_name": bug.method_name,
                    "content": revise_info.content,
                }
            )
        bug_info_json = json.dumps({"bug_info": bug_info})
        return problem_statement, bug_info_json

    def _get_response(
        self,
        current_res: List[EditBugCode],
    ) -> AgentChatResponse:
        # concat list of EditBugCode into a string
        response_str = json.dumps(
            {"revised_code": [bug.to_json() for bug in current_res]}
        )

        return AgentChatResponse(response=response_str)

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
            memory.put(ChatMessage(content=step.input, role=MessageRole.USER))
            step.step_state["is_first"] = False
        else:
            # logger.info("step input: \n" + step.input)
            memory = task.extra_state["instruct_memory"]
            memory.put(ChatMessage(content=step.input, role=MessageRole.USER))

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
        problem_statement, bug_code_input = self._process_edit_input(self._edit_input)
        # add task input to chat history
        input_chat = self._edit_formatter.format(
            problem_statement=problem_statement,
            bug_code_input=bug_code_input,
        )

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
        revised_code = self._extract_editing_step(chat_response)

        agent_response = self._get_response(revised_code)

        return self._get_task_step_response(agent_response, step, is_done=True)

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

        step_engine = EditWorker(
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

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"agent_worker": self.agent_worker}
