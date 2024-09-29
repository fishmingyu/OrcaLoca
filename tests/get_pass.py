import getpass
import os


def _get_pass(var: str):
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")


# Optional: Debug + trace calls using LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LLMCompiler"
_get_pass("LANGCHAIN_API_KEY")
_get_pass("OPENAI_API_KEY")
