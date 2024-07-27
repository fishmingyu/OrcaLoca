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


conda create -n "llama_index" python=3.10 -y
source activate "llama_index"
pkg_requirements