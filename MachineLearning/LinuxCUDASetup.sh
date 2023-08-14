# remove cuda toolkit
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
 "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"

# remove drivers
sudo apt-get --purge remove "*nvidia*" "libxnvctrl*"

# cleanup
sudo apt-get autoremove

# headers
sudo apt-get install linux-headers-$(uname -r)

# check distro version
uname -m && cat /etc/*release

# set os / arch given above
OS=ubuntu2004
arch=x86_64

# update gcc
sudo apt install --reinstall gcc

# cuda 11.7 toolkit network install; if WSL -- do not install drivers, you get drivers through windows
wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/${arch}/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-${OS}-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# post install
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo reboot

# might need symlink
sudo ln -s libcuda.so.1.1 libcuda.so.1
sudo ln -s libcuda.so.1 libcuda.so
sudo ldconfig

# Tesla K80 Change Requirement, If K80:
sudo apt-get purge nvidia-*
sudo apt install libnvidia-common-470
sudo apt install nvidia-driver-470
sudo reboot


# if WSL -- add these export paths to .bashrc with vim:
sudo vim ~/.bashrc

# set:
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.7bin:$PATH

# for bitsandbytes and wsl
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# check install
nvidia-smi
nvcc -V

# update linux
sudo apt update -y && sudo apt full-upgrade -y && sudo apt autoremove -y && sudo apt clean -y && sudo apt autoclean -y
sudo apt-get install libopenmpi-dev

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo ./Miniconda3-latest-Linux-x86_64.sh

# create env; ready for SoTA training with peft, deepspeed, and bnb
conda create --name ml python=3.10
conda activate ml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge pandas numpy transformers cudnn scikit-learn
pip install pip deepspeed accelerate sentencepiece evaluate ninja peft mpi4py bitsandbytes jupyter pyarrow --upgrade
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./


# add env to kernel list in azureml
python -m ipykernel install --user --name my_ml


