# HPA-Special-Prize

The special prize winner in the kaggle HPA competition

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 2.7 (should work fine with 3.x)
CUDA 9.0
cuddn 7.0.5
nvidia drivers v.384

## Installation
The solution is using the OpenVINO by intel. please install it before attempting to do inference

apt-get install -y --no-install-recommends \
        build-essential \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        sudo

pip3 install tensorflow numpy pandas networkx tqdm

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# below are the shell commands used in each step, as run from the top level directory
mkdir -p data/
cd data
kaggle competitions download -c human-protein-atlas-image-classificatin
mkdir test
mkdir train
cd test && unzip ../test.zip
#get external data
wget https://storage.googleapis.com/kaggle-forum-message-attachments/430860/10774/HPAv18RGBY_WithoutUncertain_wodpl.csv
python download_hpa.py
python conv_512.py


# DATA PROCESSING
# The train/predict code will also call this script if it has not already been run on the relevant data.

# MODEL BUILD: There are three options to produce the solution.
python train train/shufflenet_test_enhanced.py
# freezing to pb
git clone https://github.com/amir-abdi/keras_to_tensorflow
python3 keras_to_tensorflow/keras_to_tensorflow.py --input_model model.model --output_model frozen_model.pb
# converting to openvino
python3 /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_model.pb --input_shape [1,512,512,3] --data_type FP32


