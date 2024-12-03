import ast
from typing import List, Tuple

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM

from .code_scorer import CodeScorer
from .formatter import TokenCounter
from .log_utils import get_logger
from .tracer import FuncScore, FuncSign
from .types import CodeInfoWithClass

logger = get_logger(__name__)


class FuncSignWithClass(FuncSign):
    classname: str

    def to_str(self) -> str:
        return f"{self.classname}::{self.funcname} ({self.filename}:{self.lineno})"

    def to_codeinfo(self):
        # Return a CodeInfoWithClass object
        return CodeInfoWithClass(
            keyword=self.funcname,
            file_path=self.filename,
            class_name=self.classname,
        )


class FunctionFinder(ast.NodeVisitor):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno
        self.function_node = None
        self.classname = ""  # To store the class name if the function is inside a class
        self._current_class = None  # To track the current class during traversal

    def visit_ClassDef(self, node):
        # Get the most outer class name
        if (self._current_class is None) and (self.function_node is None):
            self.classname = node.name
            self._current_class = node.name
            self.generic_visit(node)  # Visit child nodes
            self._current_class = None
        else:
            self.generic_visit(node)

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


def get_func_content_with_class(input: FuncSign) -> tuple[str, FuncSignWithClass]:
    with open(input.filename, "r") as f:
        lines = f.readlines()

    tree = ast.parse("".join(lines))
    finder = FunctionFinder(input.funcname, input.lineno)
    finder.visit(tree)

    if not finder.function_node:
        # If no function is found, return empty content and an empty class
        return "", FuncSignWithClass(
            filename=input.filename,
            lineno=input.lineno,
            funcname=input.funcname,
            classname="",
        )

    end_lineno = finder.function_node.end_lineno
    content = "".join(lines[input.lineno - 1 : end_lineno])

    # Create a FuncSignWithClass object using the found class name
    func_sign_with_class = FuncSignWithClass(
        filename=input.filename,
        lineno=input.lineno,
        funcname=input.funcname,
        classname=finder.classname,
    )

    return content, func_sign_with_class


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


def rerank_helper(
    function_content: str, called_by: list[FuncSign]
) -> List[ChatMessage]:
    function_prompt = ChatMessage(
        role="user",
        content="<function_content>" f"{function_content}" "</function_content>",
    )
    callchain = " -> ".join([f"'{x.funcname}'" for x in called_by])
    callchain_prompt = ChatMessage(
        role="user",
        content=f"This function is traced in an issue reproducer, and its captured call chain is: {callchain}",
    )
    return [function_prompt, callchain_prompt]


def rerank_func(
    input: List[Tuple[FuncSign, FuncScore]],
    llm: LLM,
    token_counter: TokenCounter,
    problem_statement: str,
) -> List[FuncSign]:
    scorer = CodeScorer(
        llm=llm, token_counter=token_counter, problem_statement=problem_statement
    )
    output_sorted_list: List[int, FuncSign, FuncScore, int] = []
    input_message_lists: List[List[ChatMessage]] = []
    func_contents: List[str] = []
    for i, t in enumerate(input):
        func_sign, func_score = t
        func_content, func_sign_with_class = get_func_content_with_class(func_sign)
        input[i] = (func_sign_with_class, func_score)

        if not func_content:
            logger.warning("Cannot find function:")
            logger.warning(func_sign)
        func_contents.append(func_content)
        input_message_lists.append(
            rerank_helper(
                function_content=func_content,
                called_by=func_score.called_by + [func_sign],
            )
        )

    scores = scorer.score_batch(input_message_lists=input_message_lists)

    for i, t in enumerate(input):
        func_sign, func_score = t
        func_content = func_contents[i]
        llm_score = scores[i]

        logger.info(f"Func {i+1:02d}/{len(input):02d}")
        logger.info(func_sign)
        logger.info(f"LLM score: {llm_score} / 100")
        logger.info(func_content)

        llm_int_score = 100 - llm_score
        score = (
            llm_int_score
            + int(not func_score.is_same_file_with_key_parent) * 40
            + func_score.layers_from_key_parent * 20
        )
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

    output_sorted_list = [x for x in output_sorted_list if x[3] > 50]
    logger.info("----------------After filter------------------------")
    # for x in output_sorted_list:
    #    logger.info(f"Score: {x[0]}")
    #    logger.info(x[1].to_str())
    #    logger.info(f"LLM Score: {x[3]}")
    #    logger.info(x[2].get_score())
    logger.info([(x[1].funcname, x[3]) for x in output_sorted_list])

    return [x[1] for x in output_sorted_list]
