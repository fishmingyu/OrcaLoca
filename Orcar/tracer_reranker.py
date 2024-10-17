import ast
from typing import List, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

from .formatter import TokenCount, TokenCountCached, TokenCounter, TokenCounterCached
from .log_utils import get_logger
from .tracer import FuncScore, FuncSign

logger = get_logger(__name__)


class FunctionFinder(ast.NodeVisitor):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno
        self.function_node = None

    def visit_FunctionDef(self, node):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == self.funcname
            and (
                node.lineno == self.lineno
                or (
                    len(node.decorator_list)
                    and node.decorator_list[0].lineno == self.lineno
                )
            )
        ):
            self.function_node = node
        self.generic_visit(node)


def get_func_content(input: FuncSign) -> str:
    with open(input.filename, "r") as f:
        lines = f.readlines()

    tree = ast.parse("".join(lines))
    finder = FunctionFinder(input.funcname, input.lineno)
    finder.visit(tree)

    if not finder.function_node:
        return ""
    end_lineno = finder.function_node.end_lineno
    return "".join(lines[input.lineno - 1 : end_lineno])


def redirect_filepath_to_cache(
    input: List[Tuple[FuncSign, FuncScore]], cache_dir: str
) -> List[Tuple[FuncSign, FuncScore]]:
    def redirect_funcsign(func_sign: FuncSign) -> FuncSign:
        new_path = cache_dir + func_sign.filename
        return FuncSign(
            filename=new_path, lineno=func_sign.lineno, funcname=func_sign.funcname
        )

    ret: List[Tuple[FuncSign, FuncScore]] = []
    for func_sign, func_score in input:
        new_func_sign = redirect_funcsign(func_sign)
        func_score.called_by = [redirect_funcsign(fs) for fs in func_score.called_by]
        ret.append((new_func_sign, func_score))
    return ret


class FuncScorer:
    def __init__(
        self,
        llm: LLM,
        token_counter: TokenCounter,
        problem_statement: str,
    ):
        self._llm = llm
        self._token_counter: TokenCounter = token_counter
        self._enable_cache: bool = TokenCounterCached.is_cache_enabled(llm)
        self._messages_prefix: List[ChatMessage] = [
            # System prompt currently requires content to be addable with '/n' (which means: a string)
            # llama-index-llms-anthropic     0.3.4
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a python coding expert. "
                    "Your job is to score how likely a function will need to be modified "
                    "to solve a github issue. "
                    "The issue description will be presentd in 'problem_statement'. "
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="<problem_statement>"
                f"{problem_statement}"
                "</problem_statement>",
            ),
        ]
        self._order_prompt: ChatMessage = ChatMessage(
            role=MessageRole.USER,
            content=(
                "Please score how likely a function will need to be modified to solve a github issue. "
                "Please score the likeliness with an integer between 0 and 100, the higher the more likely."
                "Your output will be processed by program instead of human, "
                "so please ONLY output a single integer."
            ),
        )

        self._token_cnts: List[TokenCount] = []
        if self._enable_cache:
            self._token_counter = TokenCounterCached(llm)

    # TODO: Add function to fill cachable_fix to satisfy min length

    def score(self, function_content: str) -> int:
        function_prompt = ChatMessage(
            role="user",
            content="<function_content>" f"{function_content}" "</function_content>",
        )
        messages_prefix = self._messages_prefix
        if self._enable_cache and messages_prefix[-1].role == MessageRole.USER:
            messages_prefix[-1].additional_kwargs["cache_control"] = {
                "type": "ephemeral"
            }

        messages = self._messages_prefix + [function_prompt, self._order_prompt]
        response, cnt = self._token_counter.count_chat(llm=self._llm, messages=messages)
        logger.info(cnt)
        self._token_cnts.append(cnt)
        return int(response.message.content)

    def log_token_stats(self) -> None:
        logger.info(
            "Total cnt : "
            + str(
                sum(
                    self._token_cnts,
                    start=TokenCountCached(in_token_cnt=0, out_token_cnt=0),
                )
            )
        )


def rerank_func(
    input: List[Tuple[FuncSign, FuncScore]],
    llm: LLM,
    token_counter: TokenCounter,
    problem_statement: str,
) -> List[FuncSign]:
    scorer = FuncScorer(
        llm=llm, token_counter=token_counter, problem_statement=problem_statement
    )
    output_sorted_list: List[int, FuncSign, FuncScore, int] = []
    for i, t in enumerate(input):
        func_sign, func_score = t
        func_content = get_func_content(func_sign)
        logger.info(f"Func {i+1:02d}/{len(input):02d}")
        if func_content:
            logger.info(func_sign)
            logger.info(func_content)
        else:
            logger.warning("Cannot find function:")
            logger.warning(func_sign)
        llm_score = scorer.score(function_content=func_content)
        llm_int_score = 100 - llm_score
        score = (
            llm_int_score
            + int(not func_score.is_same_file_with_key_parent) * 40
            + func_score.layers_from_key_parent * 20
        )
        # TBD: form message list, chat, parse output into an integer
        # TBD: recalculate score & rerank
        output_sorted_list.append((score, func_sign, func_score, llm_score))
    scorer.log_token_stats()

    logger.info([x[1].funcname for x in output_sorted_list])
    logger.info("----------------Before sort-----------------------")
    # for x in output_sorted_list:
    #    logger.info(f"Score: {x[0]}")
    #    logger.info(x[1].to_str())
    #    logger.info(f"LLM Score: {x[3]}")
    #    logger.info(x[2].get_score())
    logger.info([(x[1].funcname, x[3]) for x in output_sorted_list])

    output_sorted_list.sort(key=lambda x: x[0])

    logger.info("----------------After sort------------------------")
    # for x in output_sorted_list:
    #    logger.info(f"Score: {x[0]}")
    #    logger.info(x[1].to_str())
    #    logger.info(f"LLM Score: {x[3]}")
    #    logger.info(x[2].get_score())
    logger.info([(x[1].funcname, x[3]) for x in output_sorted_list])

    return [x[1] for x in output_sorted_list]
