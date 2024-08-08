set -exuo pipefail
source ~/.bashrc

function pkg_requirements {
    pip3 install llama-index-core
    pip install pydantic
    pip install llama-index-embeddings-openai 
    pip install llama-index-agent-llm-compiler
    pip install llama-index-llms-openai
    pip install numexpr
}


conda create -n "test" python=3.10 -y
source activate "test"
pkg_requirements