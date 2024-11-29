## Orca Runtime for LLM

Installation: pip install -e .

Executing prompt: orcar execute --docker 'Run ls'

Run a benchmark: orcar benchmark -p -c test -f 'astropy__astropy-14182'

Run evaluation:

python artifact/evaluate_output.py --max_workers 1 --orcar_root_path '.' --run_id test

or run evaluation for specific insts

python artifact/evaluate_output.py --max_workers 3 --orcar_root_path '.' --run_id test_3_insts --instance_ids astropy__astropy-14995 django__django-12983 django__django-12700
