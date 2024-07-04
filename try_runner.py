from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker

# construct OpenAIAgent from tools
openai_step_engine = OpenAIAgentWorker.from_tools(tools, llm=llm, verbose=True)
agent = AgentRunner(openai_step_engine)

# create task
task = agent.create_task("What is (121 * 3) + 42?")

# execute step
step_output = agent.run_step(task)

# if step_output is done, finalize response
if step_output.is_last:
    response = agent.finalize_response(task.task_id)

# list tasks
task.list_tasks()

# get completed steps
task.get_completed_steps(task.task_id)

print(str(response))
