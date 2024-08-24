from Orcar import SearchAgent
from llama_index.llms.openai import OpenAI

def test_search_agent():
    llm = OpenAI(model="gpt-4o")
    agent = SearchAgent(repo_path="./test_repo", llm=llm, verbose=True)
    response = agent.chat("""There are some bugs in the test_repo code. Below is the relevant info:
                          Traceback (most recent call last):
                            File "/home/zhongming/IntelliCopilot/tests/test_repo/a.py", line 15, in <module>
                                print(add(2, 3))
                            File "/home/zhongming/IntelliCopilot/tests/test_repo/a.py", line 6, in add
                                return a + b + c
                            NameError: name 'c' is not defined
                           Help me locate the related function)
                          """)
    print(response)

if __name__ == "__main__":
    test_search_agent()