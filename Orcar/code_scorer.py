import asyncio
import time
from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

from .formatter import TokenCount, TokenCountCached, TokenCounter, TokenCounterCached
from .log_utils import get_logger

logger = get_logger(__name__)

CODE_SCORER_SYSTEM_PROMPT = (
    "You are a python coding expert. "
    "Your job is to score how likely a piece of code will need to be modified "
    "to solve a github issue. "
    "The issue description will be presented in 'problem_statement'. "
)

CODE_SCORER_ORDER_PROMPT = (
    "Please score how likely this piece of code will need to be modified to solve a github issue. "
    "Please score the likeliness with an integer between 0 and 100, the higher the more likely."
    "Your output will be processed by program instead of human, "
    "so please ONLY output a single integer."
)


class CodeScorer:
    """
    Give relevance score between code piece and solving issue.
    """

    def __init__(
        self,
        llm: LLM,
        problem_statement: str,  # inst['problem_statement']
        token_counter: TokenCounter | None = None,
    ):
        self._llm = llm
        self._token_counter: TokenCounter = (
            token_counter if token_counter else TokenCounter(llm)
        )
        self._enable_cache: bool = TokenCounterCached.is_cache_enabled(llm)
        self._messages_prefix = [
            # System prompt currently requires content to be addable with '/n' (which means: a string)
            # llama-index-llms-anthropic     0.3.4
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=CODE_SCORER_SYSTEM_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="<problem_statement>"
                f"{problem_statement}"
                "</problem_statement>",
            ),
        ]
        self._order_prompt = ChatMessage(
            role=MessageRole.USER, content=CODE_SCORER_ORDER_PROMPT
        )

        self._token_cnts: List[TokenCount] = []
        if self._enable_cache:
            logger.info(f"Cache is enabled for llm type {type(llm)}")
            self._token_counter = TokenCounterCached(llm)

    # TODO: Add function to fill cachable_fix to satisfy min length

    async def _score_batch_async(self, chat_inputs: List[List[ChatMessage]]):
        tasks = [
            self._token_counter.count_achat(llm=self._llm, messages=chat_input)
            for chat_input in chat_inputs
        ]
        return await asyncio.gather(*tasks)

    def score_batch(self, input_message_lists: List[List[ChatMessage]]) -> List[int]:
        """
        Score each function by LLM in parallel.

        input_message_lists: List of input_message_list you wish to score
            input_message_list can contain code snippet, call chain,
            or other info you believe LLM should know
        """
        messages_prefix = self._messages_prefix
        if self._enable_cache and messages_prefix[-1].role == MessageRole.USER:
            # Specific way to pass this "type" to anthropic API
            # llama-index-llms-anthropic     0.3.4
            messages_prefix[-1].additional_kwargs["cache_control"] = {
                "type": "ephemeral"
            }
        chat_inputs = [
            messages_prefix + x + [self._order_prompt] for x in input_message_lists
        ]

        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no current event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        start_time = time.time()
        # add a buffer avoid too many requests
        results = []
        for i in range(0, len(chat_inputs), 10):
            results += loop.run_until_complete(
                self._score_batch_async(chat_inputs[i : i + 10])
            )

        logger.info(f"Total batch chat time: {time.time() - start_time:.2f}s")

        ret: List[int] = []
        for response, cnt in results:
            logger.info(cnt)
            self._token_cnts.append(cnt)
            ret.append(int(response.message.content))
        return ret

    def clear_sum_cnt(self) -> None:
        self._token_cnts = []

    def get_sum_cnt(self, display: bool = False) -> TokenCount:
        if isinstance(self._token_counter, TokenCounterCached):
            sum_cnt = sum(
                self._token_cnts,
                start=TokenCountCached(in_token_cnt=0, out_token_cnt=0),
            )
            if display:
                logger.info(f"{'Total Scorer cached cnt':<25}: " + str(sum_cnt))
            sum_cnt = self._token_counter.equivalent_cost(sum_cnt)
        else:
            sum_cnt = sum(
                self._token_cnts,
                start=TokenCount(in_token_cnt=0, out_token_cnt=0),
            )
        return sum_cnt

    def log_token_stats(self) -> None:
        sum_cnt = self.get_sum_cnt(display=True)
        logger.info(
            (
                f"{'Total Scorer cnt':<25}: "
                f"in {sum_cnt.in_token_cnt:>6} tokens, "
                f"out {sum_cnt.out_token_cnt:>6} tokens"
            )
        )
