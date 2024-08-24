from pydantic import BaseModel, Field
from typing import List

from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage


class CodeLocationInfo(BaseModel):
    """Location info of code, extracted from natural language description"""

    file_path: str | None = Field(
        description="""
            The path of the file containing the code. Can be relative or absolute path.
            Levels of path should only be spliced with '/', '\', not ' '.
            """
    )
    function_name: str | None = Field(
        description="Name of the function containing the code"
    )
    line_of_code: int | None = Field(
        description="Line number in file where the code lies"
    )


class RawIssueInfo(BaseModel):
    """Information extracted from human reported github issue"""

    summary: str = Field(description="""Summary of issue""")
    issue_reproducer: str | None = Field(
        description="""
            Code snippet to reproduce the issue. Should be a python code snippet that can be directly runned.
            \n should be used for new line, 4 spaces should be used for indentation.
            If file creation is necessary, python file IO should be used.
            If the reproducer is mentioned in interactive mode, the code should be extracted and spliced into a snippet.
            Code shouldn't be inferred from natural language description.
            """
    )
    related_code_snippets: List[CodeLocationInfo] = Field(
        description="""
            List of mentioned suspicious codes snippets.
            Each element should contain different info of ONE code snippet.
            ALL code snippets mentioned in issue description should be extracted, like in traceback, warning, etc.
            """
    )


class ReproducerInfo(BaseModel):
    """Information extracted from reproducer execution result"""

    is_issue_reproduced: bool = Field(
        description="""
        Whether the issue is successfully reproduced.
        Notice that if the reproduce snippet failed with the same error described in issue,
        it is considered as a successful reproduce.
        """
    )
    related_code_snippets: List[CodeLocationInfo] = Field(
        description="""
            List of mentioned suspicious codes snippets in reproducer execution result.
            Set to [] if is_success == False.
            Each element should contain different info of ONE code snippet.
            ALL code snippets mentioned in issue description should be extracted, like in traceback, warning, etc.
            """
    )


class IssueInfo(BaseModel):
    """Information produced by extractor"""

    summary: str = Field(description="""Summary of issue""")
    issue_reproducer_info: ReproducerInfo = Field(description="""Reproducer of issue""")
    related_code_snippets: List[CodeLocationInfo] = Field(
        description="""
            List of mentioned suspicious codes snippets.
            Each element should contain different info of ONE code snippet.
            ALL code snippets mentioned in issue description should be extracted, like in traceback, warning, etc.
            """
    )


def get_extractor_function(llm: OpenAI) -> OpenAIPydanticProgram:
    prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert python developer, mastering at summarizing and extracting from github issues."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Please extract the RawIssueInfo from this issue description: \n"
                    "------\n"
                    "{issue_description}\n"
                    "------"
                ),
            ),
        ]
    )
    return OpenAIPydanticProgram.from_defaults(
        output_cls=RawIssueInfo, llm=llm, prompt=prompt
    )


def get_reproducer_function(llm: OpenAI) -> OpenAIPydanticProgram:
    prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert python developer, mastering at reproducing github issues."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Please check the given issue and reproduce_history (containing reproducer snippet and exection log): \n"
                    "If the snippet successfully reproduced the issue, extract code info from it.\n"
                    "<issue_description>\n"
                    "{issue_description}\n"
                    "</issue_description>\n"
                    "<reproduce_history>\n"
                    "{reproduce_history}\n"
                    "</reproduce_history>"
                ),
            ),
        ]
    )
    return OpenAIPydanticProgram.from_defaults(
        output_cls=ReproducerInfo, llm=llm, prompt=prompt
    )
