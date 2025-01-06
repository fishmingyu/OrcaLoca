import os

# cd to the root of the project
os.chdir("../../")
# cd to third_party/OpenHands
os.chdir("third_party/OpenHands")

# ./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/claude-3-5-sonnet-20241022_maxiter_60_N_v0.17.0-no-hint-run_1/output.jsonl
os.system(
    "./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/claude-3-5-sonnet-20241022_maxiter_60_N_v0.17.0-no-hint-run_1/output.jsonl"
)
