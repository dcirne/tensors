## Overview of the progress thus far installing TensorFlow on Linux and trying to build a simple sample code.

After installing VirtualBox and Ubuntu on your system:

Update the system
```bash
sudo apt update
sudo apt upgrade
sudo apt autoremove
```

Install the dependencies
```bash
sudo apt install autoconf automake libtool curl make g++ 
sudo apt install unzip git openjdk-8-jdk pkg-config git-lfs 
sudo apt install python-numpy swig python-dev python-wheel python-pip
sudo apt-get install python-tk
pip install tensorflow
pip install matplotlib
```

Configure git

Generate and configure the ssh key: https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/#platform-linux

Setup git: https://help.github.com/articles/set-up-git/

Git-LFS: git lfs install

Install bazel
```bash
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt update
sudo apt install bazel
```

Clone TensorFlow
```bash
mkdir work
cd work
git clone git@github.com:tensorflow/tensorflow.git
```

Download the TensorFlow dependencies
```bash
cd ~/work/tensorflow
./tensorflow/contrib/makefile/download_dependencies.sh
```

Build protobuf
```bash
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure
make clean
make
sudo make install
sudo ldconfig
```

Build TensorFlow

>If you tried to build before, or there is a bazel server running
>```bash
>cd ~/work/tensorflow
>bazel clean
>bazel shutdown
>```

```bash
./configure # Say no to everything
bazel build tensorflow:libtensorflow.so
```

Generate the ops classes
```bash
bazel build tensorflow/cc:cc_ops
```

### Fit_curve

Train model and generate graph
```bash
cd ~/work/tensors/py
python fit_curve
```

Visualize with TensorBoard
```bash
cd ~/work/tensors
tensorboard --logdir=./models
```

Open a browser and point to `localhost:6006`

Run the C++ counterpart
```bash
cd ~/work/tensors
bazel run //src:fit_curve
```
