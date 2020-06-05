#!/bin/sh
apt-get update
pip3 install tensorflow-gpu==1.12.0
pip3 install bunch==1.0.1
pip3 install tqdm==4.28.1
pip3 install opencv-python==3.4.3.18
pip3 install numpy==1.13.1
pip3 install git+https://github.com/epigramai/tfserving-python-predict-client.git
#etc.