set -exuo pipefail
source ~/.bashrc

function pkg_requirements {
    pip3 install llama-index-core
    pip install pydantic
}


conda create -n "llama_index" python=3.11 -y
source activate "llama_index"
pkg_requirements