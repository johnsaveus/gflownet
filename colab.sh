#!/bin/bash

sudo apt-get update -y
sudo apt-get install python3.11 python3.11-dev python3.11-distutils libpython3.11-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

sudo apt-get install python3-pip

git clone -b setup_chemprop https://github.com/johnsaveus/gflownet.git

cd gflownet

pip install --upgrade pip setuptools wheel
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
cd src/gflownet/tasks
