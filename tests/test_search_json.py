import re
import json

output = """
{                                                                          
             "API_calls": [                                                         
                 "search_func('add')",
                 "api_call_2(args)",                                               
             ],                                                                     
             "bug_locations": [                                                     
                 {                                                                  
                     "file": "/home/zhongming/IntelliCopilot/tests/test_repo/a.py", 
                     "function": "add",                                             
                     "content": "def add(a, b):\n    return a + b + c"              
                 },
                 {                                                                  
                     "file": "/home/zhongming/IntelliCopilot/tests/test_repo/a.py", 
                     "function": "add",                                             
                     "content": "def add(a, b):\n    return a + b + c"              
                 }                                                                       
             ]                                                                      
         }  
"""

## Extract API calls
# replace \\n with \n
api_calls_content = re.search(r'"API_calls": \[(.*?)\]', output, re.DOTALL).group(1)
api_calls = re.findall(r'"(.*?)"', api_calls_content)
print(api_calls)

bug_locations_match = re.search(r'"bug_locations": \[(.*?)\]', output, re.DOTALL)
# add back [ and ] to the bug_locations string

if not bug_locations_match:
    raise ValueError("bug_locations section not found or incorrectly formatted.")
bug_locations_str = bug_locations_match.group(1)
bug_locations_str = "[" + bug_locations_str + "]"

def escape_newlines_in_json_strings(json_str):
    # Find all strings in the JSON and replace \n within them
    def replace_newline(match):
        # Replace \n with \\n inside the string
        return match.group(0).replace('\n', '\\n')
    
    # Regular expression to match strings in the JSON
    json_str = re.sub(r'\"(.*?)\"', replace_newline, json_str, flags=re.DOTALL)
    return json_str

# bug_locations_str could contain \n within the code snippet
formatted_str = escape_newlines_in_json_strings(bug_locations_str)

bug_locations = json.loads(formatted_str)

print(bug_locations)