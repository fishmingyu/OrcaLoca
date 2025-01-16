# Running Agentless editor as integration of Orcar
Our Orcar system has high modularity, which allows it to integrate part of other system.

## Prerequisite

Before running Agentless as integration of Orcar, please make sure you have either followed the instructions under OrcarLLM/README.md, or have the files below prepared:
1. evaluation/output.json
2. evaluation/dependency_output.json

## Run
```shell
cd evaluation/orcar_agentless
python prepare_agentless.py
cd ../../thirdparty/Agentless
```

Then go through instructions in thirdparty/Agentless/README_orcar.md
