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

# Install CUDA 11.7 and cudnn 8.5.0.96
mkdir installspace
cd /installspace && wget -q https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run -O cuda_11.7.0_515.43.04_linux.run && \
    sudo bash ./cuda_11.7.0_515.43.04_linux.run --toolkit --silent && \
    rm -f cuda_11.7.0_515.43.04_linux.run
cd /installspace && wget -q https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
    tar xJf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz && \
    cd cudnn-linux-x86_64-8.5.0.96_cuda11-archive && \
    sudo cp include/* /usr/local/cuda-11.7/include && \
    sudo cp lib/* /usr/local/cuda-11.7/lib64 && \
    sudo ldconfig && \
    cd .. && rm -rf cudnn-linux-x86_64-8.5.0.96_cuda11-archive && rm -f cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

# post install
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo reboot

# if WSL -- add these export paths to .bashrc with vim:
sudo vim ~/.bashrc

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


