Yalm (Yet Another Language Model) is an LLM inference implementation in C++/CUDA, using no libraries except for I/O.

# Instructions

Yalm requires a computer with a C++20-compatible compiler and the CUDA toolkit (including `nvcc`) to be installed. You'll also need a directory containing LLM safetensor weights and configuration files in huggingface format, which you'll need to convert into a `.yalm` file. Follow the below to download Mistral-7B-v0.2, build `yalm`, and run it:

```
# install git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get -y install git-lfs
# download Mistral
git clone git@hf.co:mistralai/Mistral-7B-Instruct-v0.2
# clone this repository
git clone git@github.com:andrewkchan/yalm.git

cd yalm
pip install -r requirements.txt
python convert.py --dtype fp16 mistral-7b-instruct-fp16.yalm ../Mistral-7B-Instruct-v0.2/
./build/main mistral-7b-instruct-fp16.yalm -i "What is a large language model?" -m c
```