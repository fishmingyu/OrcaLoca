# Running Agentless editor as integration of Orcar
Our Orcar system has high modularity, which allows it to integrate part of other system.
For instance, to run Orcar Localization + Agentless Edition, we just need to:
1. Run orcar localization evaluation script (As described in OrcarLLM/README.py)
2. Move to evaluation and confirm output.json & dependency_output.json is already there
3. Move to evaluation/orcar_agentless and run prepare_agentless.py
4. Move to thirdparty/Agentless and go through instructions in README_orcar.md
