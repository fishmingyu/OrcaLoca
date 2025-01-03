import os

# cd to the root of the project
os.chdir("../../")
# cd to third_party/OpenHands
os.chdir("third_party/OpenHands")
# run the command DEBUG=1 ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.claude HEAD CodeActAgent
os.system(
    "DEBUG=1 ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh llm.claude HEAD CodeActAgent"
)
