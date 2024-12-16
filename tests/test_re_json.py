import json

# import re

log_string = r"""
{
    "observation_feedback": "The bug is in the _line_type function where the regular expression _command_re = r\"READ [TS]ERR(\\s+[0-9]+)+\" strictly matches uppercase 'READ'. The _get_tables_from_qdp_file function processes the command lines but relies on _line_type for command detection. While err_specs dictionary uses lowercase keys, the initial command detection fails due to case sensitivity.",
    "potential_bug_locations": [
        {
            "file_name": "astropy/io/ascii/qdp.py",
            "class_name": "",
            "method_name": "_line_type"
        },
        {
            "file_name": "astropy/io/ascii/qdp.py",
            "class_name": "",
            "method_name": "_get_tables_from_qdp_file"
        }
    ],
    "new_search_actions": [
        {
            "action": "fuzzy_search",
            "action_input": {
                "query": "QDPReader"
            }
        },
        {
            "action": "exact_search",
            "action_input": {
                "query": "_get_type_from_list_of_lines",
                "file_path": "astropy/io/ascii/qdp.py"
            }
        }
    ]
}
"""


def load_with_escape(input_text: str) -> dict:
    # if input_text contains \s, replace it with __ESCAPED_S__
    # input_text = r"{}".format(input_text)
    input_text = input_text.replace(r"\\s", "__ESCAPED_S__")

    # Parse JSON
    data = json.loads(input_text)

    # Replace escaped characters only for observation_feedback
    data["observation_feedback"] = data["observation_feedback"].replace(
        "__ESCAPED_S__", r"\s"
    )

    return data


if __name__ == "__main__":
    print(load_with_escape(log_string))
