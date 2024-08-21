from Orcar import SearchAgent

def test_search_agent():
    agent = SearchAgent(repo_path="./test_repo", llm="gpt-4o")
    response = agent.chat("""There are some bugs in the test_repo code. Below is the relevant info:
                          Traceback (most recent call last):
                            File "/home/zhongming/IntelliCopilot/tests/test_repo/a.py", line 14, in <module>
                                print(add(2, 3))
                            File "/home/zhongming/IntelliCopilot/tests/test_repo/a.py", line 5, in add
                                return a + b + c
                            NameError: name 'c' is not defined
                           Help me locate the related function)
                          """)
    print(response)

if __name__ == "__main__":
    test_search_agent()