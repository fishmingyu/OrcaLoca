STEP_EXAMPLE = {
    "observation_feedback": "observation",
    "potential_bug_locations": [
        {
            "file_path": "path/to/file",
            "class_name": "class_name",
            "method_name": "function_name",
        },
        {
            "file_path": "path/to/file",
            "class_name": "class_name",
            "method_name": "function_name",
        },
    ],
    "new_search_actions": [
        {"action": "search_class", "action_input": {"class_name": "str"}},
        {
            "action": "search_method_in_class",
            "action_input": {"class_name": "str", "method_name": "str"},
        },
        {"action": "search_callable", "action_input": {"query_name": "str"}},
    ],
}

BUG_OUTPUT = {
    "bug_locations": [
        {
            "file_path": "path/to/file",
            "class_name": "class_name",
            "method_name": "function_name",
        },
        {
            "file_path": "path/to/file",
            "class_name": "class_name",
            "method_name": "function_name",
        },
    ]
}

SEARCH_SYSTEM_HEADER = r"""
You are a professional software engineer that use API calls to report bug code snippets from a text into json format.
You need to extract where are the bug locations by analyzing the text.
The given text contains the problem statement and the code snippets.
There are some API calls that you can use to extract the information.
The API calls include:
{tool_desc}

<TASKS>
Everytime you will do the following things:

1. Provide the observation based on given input:
Everytime we will provide a new search result in tag <New Info>.
It may contain the disambiguation info if the search action is related to multiple classes or methods.
Also, previous search result will be provided in tag <Search Result>. You need to analyze the new search result based on the previous one, and provide the observation
based on whole context.
2. Think about where the bug might be in the code by the whole given context(including all Search Result), and provide the potential bug locations. Potential here means the most possible locations up to current context.
3. Check whether it contains any class, method or function you need to further search. Especially, if disambiguation info is provided, you need to search for the specific class or method.
Plan the new_search_actions based on the current context. You can use the given API calls to search for the bug locations.
You can put multiple actions in the new_search_actions list. Be sure to use arguments in the tool description.
If you make sure the context is enough to answer the question, you can keep the new_search_actions list empty.

Conclusion is a final standalone step to provide the final bug locations when nothing else to search. Please keep in mind to
follow the instruction "Now let's come to a conclusion. ".
</TASKS>

<OUTPUT FORMAT>
1. Regular Step Format:
    Provide your answer in a clear JSON structure like this,
    {step_format}
    Make sure each API call is written as a valid python expression and code_snippet is a valid python string.
    In potential_bug_locations, you should provide the file path, class name and method name.
    It's not the final answer, just a hint for possible bug locations.
    If a method does not belong to any class, set class to empty string.
    You can provide multiple actions in the new_search_actions. DO NOT add any title or description.
2. Conclusion Format:
    After no input actions in search queue, provide the final bug locations in JSON structure like this.

    {bug_locations}
    DO NOT generate observation or new_search_actions in the conclusion step.
    DO NOT mix it with any title or description. If a method does not belong to any class, set class to empty string.
</OUTPUT FORMAT>
"""

BUG_CODE_INPUT = {
    "bug_info": [
        {
            "file_path": "path/to/file",
            "func_name": "function_name",
            "range": "line numbers",
        },
        {
            "file_path": "path/to/file",
            "func_name": "function_name",
            "range": "line numbers",
        },
    ]
}

EDIT_OUTPUT = {
    "feedback": "observation",
    "action_input": {"command": "view", "path": "str"},
}

EDIT_SYSTEM_HEADER = r"""
You are a professional software engineer that uses editor tools to revise code snippets based on the bug report.
Your task is to analyze the problem statement and the code snippets, and then decide which commands to use to revise the code snippets.
Here is the API call you can use to revise the code snippets:
{tool_desc}
You can view the whole file and references location if you find the context is not enough to revise the code snippets.
We will provide you with the problem statement, the code snippets that need to be revised, and some of the references.
<INPUT>
You will receive the code snippets that need to be revised in the following JSON format:
{bug_code_format}
Also, we will provide you a hint observation in tag <Observation> based on the given input.
</INPUT>

<OUTPUT FORMAT>
Please also provide the output in a clear JSON structure like this:
{edit_output_format}
Put the observation in the feedback field. The feedback should be a string.
Respond to the feedback with "PATCH COMPLETE" if you think you fixed the issue, and leave the action_input as an empty dict.
The action_input should be a dictionary of editor's arguments.
Each time you can only use one command and provide one feedback.
DO NOT add any title or description outside the JSON structure.
</OUTPUT FORMAT>
"""

EDIT_REQUIREMENTS = r"""
The revised code snippets should respect the requirements below:
1. The revised code snippets should fix the issue mentioned in the problem statement.
2. The code containing any error messages should be deterministic.
"""

EXTRACT_FORMATS = {
    "slice": {
        "traceback_warning_log_slice": "log_slice_string",
        "issue_reproducer_slice": "code_slice_string",
        "source_code_slice": "code_slice_string",
    },
    "parse": {
        "code_info_list": [
            {"keyword": "class_or_function_name_1", "file_path": ""},
            {"keyword": "class_or_function_name_2", "file_path": "file_path_2"},
        ]
    },
    "judge": {"is_successful": True, "fixed_reproduce_snippet": "code_string"},
    "summarize": {
        "summary": "summary_string",
        "code_info_list": [
            {"keyword": "class_or_function_name_1", "file_path": ""},
            {"keyword": "class_or_function_name_2", "file_path": "file_path_2"},
        ],
    },
}

EXTRACT_FIELDS = {
    "slice": """
<field>
    traceback_warning_log_slice: Traceback or warning log. Set to '' if not found.
</field>
<field>
    issue_reproducer_slice: Code snippet to reproduce the issue. Should be a python code snippet that can be directly runned.
            \n should be used for new line, 4 spaces should be used for indentation.
            If the reproducer is mentioned in interactive mode, the code should be extracted and parsed into an .py file.
            For example, '>>> ' should never be used in an .py file, and the output of interactive shell should also be excluded.
            Code shouldn't be inferred from natural language description. Set to '' if not found.
</field>
<field>
    source_code_slice: Code referenced in the issue which comes from the source repo. Should have python code only.
            DO NOT label code as this category UNLESS EXPLICITE words are found,
            saying that it is COPIED / REFERENCED from the source repo.
            Shouldn't overlap with traceback_warning_log_slice or issue_reproducer_slice.
            Set to '' if no code satisfies this requirement.
</field>
""",
    "parse": """
<field>
    keyword: the name of the class, function, method or global variable where the suspicious code lies in.
            Should be a single word, not spliced with dot.
</field>
<field>
    file_path: The path of the file containing the code. Can be relative or absolute path.
            Levels of path should only be spliced with slash or backslash, not space.
            Specially, python import style path should be parsed as:
            1. dot replaced with slash;
            2. add .py suffix if no suffix is found.
            For example, "pvlib.bifacial.pvfactors" should be interpreted as "pvlib/bifacial/pvfactors.py"
            Set to '' if cannot find path.
</field>
<field>
    code_info_list: list of (keyword, file_path). All keywords mentioned should be extracted to the list.
</field>
""",
    "judge": """
<field>
    is_successful: whether the reproduce_snippet successfully reproduced the issue, based on the reproducer_log it generated.
            Note that 'successfully reproduce' means the similar phenomenon is observed;
            It does not necessarily means the snippet finished without error
            (Getting the same error reported in issue means reproduction is successful)
</field>
<field>
    fixed_reproduce_snippet: If is_successful is true, this should be set to empty string;
            Otherwise, generate a fixed reproduce snippet based on previous snippet, problem description and previous log.
</field>
""",
    "summarize": """
<field>
    summary: Summary in natural language. Requirements include:
            1. Describe the issue;
            2. Suggest the methods/classes/functions/files that following agents should examine;
            3. Be within 50 words.
</field>
<field>
    code_info_list: list of (keyword, file_path).
            All keywords mentioned in natural language (not code snippet or traceback) should be extracted to the list.
</field>
<field>
    keyword: the name of the class or function where the suspicious code lies in.
            Should be a single word, not spliced with dot.
</field>
<field>
    file_path: The path of the file containing the code. Can be relative or absolute path.
            Levels of path should only be spliced with slash or backslash, not space.
            Specially, python import style path should be parsed as:
            1. dot replaced with slash;
            2. add .py suffix if no suffix is found.
            For example, "pvlib.bifacial.pvfactors" should be interpreted as "pvlib/bifacial/pvfactors.py"
            Set to '' if cannot find path.
</field>
""",
}

EXTRACT_EXAMPLES = {
    "slice": {
        "repo_name": "marshmallow-code/marshmallow",
        "input_description": """
3.0: DateTime fields cannot be used as inner field for List or Tuple fields

`DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`.

```python
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

MySchema()
```

Traceback:
```
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
AttributeError: 'List' object has no attribute 'opts'
```

It seems like it's treating the parent field as a Schema without checking that it is indeed a schema.
""",
        "example_output": {
            "traceback_warning_log_slice": """
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
AttributeError: 'List' object has no attribute 'opts'
""",
            "issue_reproducer_slice": """
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

MySchema()
""",
            "source_code_slice": "",
        },
    },
    "parse": {
        "traceback": {
            "repo_name": "marshmallow-code/marshmallow",
            "input_description": """
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py", line 636, in _bind_to_schema
    self.inner._bind_to_schema(field_name, self)
AttributeError: 'List' object has no attribute 'opts'
""",
            "example_output": {
                "code_info_list": [
                    {"keyword": "<module>", "file_path": "test-mm.py"},
                    {
                        "keyword": "__init__",
                        "file_path": "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py",
                    },
                    {
                        "keyword": "_bind_to_schema",
                        "file_path": "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py",
                    },
                ]
            },
        },
        "code": {
            "repo_name": "marshmallow-code/marshmallow",
            "input_description": """
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

MySchema()
""",
            "example_output": {
                "code_info_list": [
                    {"keyword": "fields", "file_path": ""},
                    {"keyword": "Schema", "file_path": ""},
                ]
            },
        },
    },
    "summarize": {
        "repo_name": "marshmallow-code/marshmallow",
        "input_description": """
3.0: DateTime fields cannot be used as inner field for List or Tuple fields

`DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`.

```python
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

MySchema()
```

Traceback:
```
Traceback (most recent call last):
  File "test-mm.py", line 8, in <module>
    MySchema()
  File "/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py", line 383, in __init__
    self.fields = self._init_fields()
AttributeError: 'List' object has no attribute 'opts'
```

It seems like it's treating the parent field as a Schema without checking that it is indeed a schema.
""",
        "example_output": {
            "summarize": """
In marshmallow 3.0, using DateTime fields as inner fields in List or Tuple containers triggers an AttributeError.
The error occurs because List is mistakenly treated as a schema.
Examine the fields.List, fields.DateTime, and _init_fields methods in schema.py for debugging.
""",
            "code_info_list": [
                {"keyword": "DateTime", "file_path": ""},
                {"keyword": "Schema", "file_path": ""},
                {"keyword": "opts", "file_path": ""},
            ],
        },
    },
}

EXTRACT_PROMPTS = {
    "header": """
You are an expert python developer, mastering at summarizing and extracting from github issues.
""",
    "example": r"""
<repo_name>{example_repo_name}<repo_name>
<example_input_description>
{example_input_description}
</example_input_description>
<example_output>
{example_output}
</example_output>
""",
    "slice": r"""
Your task is to slice strings from human reported github issue.
Every slice shouldn't overlap with another slice. Non-existanct slice should be set to ''.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

An example is given below:
{example}

Below is the real task for you to solve:
<repo_name>{repo_name}</repo_name>
{input_description}
""",
    "parse": r"""
Your task is to extract python code keywords and the filepath they belong to (if exist) from human reported github issue.
Non-existanct filepath should be set to ''.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

An example is given below:
{example}

Below is the real task for you to solve:
<repo_name>{repo_name}</repo_name>
<input_description>
{input_description}
</input_description>
""",
    "judge": r"""
Your task is to judge whether an input github issue is successfully reproduced,
based on the reproducer_log generated by a reproducer snippet;
If reproduce didn't succeed, try to generate a fixed reproduced snippet.

Some examples of judgement include:
1. SUCCESS if (the exactly same error message) from input_description is found in reproducer_log;
1. FAILURE if the error message from input_description is different or irrelevant from the one found in reproducer_log;
1. SUCCESS if (the same printed output) from input_description is found in reproducer_log;
1. FAILURE if the reproducer in input_description is expected to have output (error or printed log) but reproducer_log is empty;
1. FAILURE if the reproducer in input_description is expected to raise an error, but no error found from reproducer_log;
1. FAILURE if the reproducer in input_description is not expected to raise any errors, but 1 or more errors are found from reproducer_log;
1. FAILURE if the input_description describes different output for expected and problematic behavior, but the reproducer_log matches with expected one;

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

Below is the real task for you to solve:
<repo_name>{repo_name}</repo_name>
<input_description>
{input_description}
</input_description>
<reproducer_snippet>
{reproducer_snippet}
</reproducer_snippet>
<reproducer_log>
{reproducer_log}
</reproducer_log>
""",
    "summarize": r"""
Your task is to summarize a human reported github issue in natural language.

Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)

The meanings of each field are:
{output_fields}

An example is given below:
{example}

Below is the issue for you to summarize:
<repo_name>{repo_name}</repo_name>
<input_description>
{input_description}
</input_description>
""",
}

FMT_CONTROL_PROMPT = r"""
Your output should strictly follow the format below.
{output_format}
DO NOT SPEAK ANY REDUNDANT WORDS (like 'json', 'output', etc.)
"""
