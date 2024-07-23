set -exuo pipefail
INSTALL_PATH="/home/user"
conda="Miniconda3-latest-Linux-x86_64.sh"
if [ "${BUILD_PLATFORM}" == "x86_64" ]; then
    conda="Miniconda3-latest-Linux-x86_64.sh"
elif [ "${BUILD_PLATFORM}" == "aarch64" ]; then
    conda="Miniconda3-latest-Linux-aarch64.sh"
else
    echo "Unsupported platform: '${BUILD_PLATFORM}'"
	exit 1
fi

cd ${INSTALL_PATH}

wget https://repo.anaconda.com/miniconda/${conda}

if file $conda | grep -q "shell script"; then
    chmod 777 $conda
    bash $conda -b -p /home/user/miniconda3
else
    echo "Downloaded file is not a valid shell script."
    exit 1
fi

rm ${conda}
echo "export PATH="/home/user/miniconda3/bin:$PATH"" > ~/.bashrc
source ~/.bashrc
conda --version
