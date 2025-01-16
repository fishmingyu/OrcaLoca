## Orca Runtime for LLM

Installation: pip install -e .

Run a benchmark: python evaluation/run.py --final_stage extract --instance_ids astropy__astropy-12907 astropy__astropy-6938

Run evaluation:

python artifact/evaluate_output.py --max_workers 1 --orcar_root_path '.' --run_id test

or run evaluation for specific insts

python artifact/evaluate_output.py --max_workers 3 --orcar_root_path '.' --run_id test_3_insts --instance_ids astropy__astropy-14995 django__django-12983 django__django-12700
