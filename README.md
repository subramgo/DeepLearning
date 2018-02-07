# Tasks

## Real-time Movement Detection

## Object Bounding

## Object Detection / Localizaiton

using DLUtils/downloadimages.py, download images "boat in sea"
stored in data/sea/boats/.

* Image annotation.

outside the virutalenvironment
brew install qt qt4 
brew install pyqt  # qt4 is deprecated
pip install labelme

# Raspberry Pi and Docker

## Raspberry Pi OS

### SDHC Preparation

Install an OS image using `etcher`

  * [openSUSE Linux](https://en.opensuse.org/HCL:Raspberry_Pi3)
    * 64-bit runtime
  * [Ubuntu Mate](https://ubuntu-mate.org/raspberry-pi/)
    - current Ubuntu desktop with `apt` support
    - preconfigured for `armhf` base packages
    * only 32-bit runtime
  * [Ubuntu Desktop](https://www.ubuntu.com/download)


## Python+Ubuntu+RaspberryPi Packages

Don't run `pip install -r requirements.txt` as it will take forever on the large packages.

### PyQT

  * Must be installed by `apt`
    * `sudo apt install build-essential python3-dev libqt4-dev`

### H5PY

  * Install `libhdf5-dev` first!
    * `sudo apt install libhdf5-dev -y`
  * Takes a long time to compile and install
    * try verbose mode `pip3 install h5py -vvv`

### Tensorflow

  1. [Build Protobuf and Bazel from source](http://cudamusing.blogspot.com/2015/11/building-tensorflow-for-jetson-tk1.html)
  2. [Build Tensorflow from source](https://www.tensorflow.org/install/install_sources)
  
NOPE:
  * [Cross-Compile](https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi/)
    1. get the [latest nightly build](http://ci.tensorflow.org/view/Nightly/job/nightly-pi-zero-python3/lastSuccessfulBuild/artifact/output-artifacts/)
    2. change 'cp34' to read 'cp35'
    3. `sudo pip3 install tensorflow-...`

### OpenCV

  1. Use PIP
    * `pip3 install opencv-python`
  2. Use APT
    * `sudo apt install python-opencv`
  3. [Build from source](https://opencv.org)

## Install Docker

### Mac

    brew install docker


### Ubuntu

    sudo apt install docker.io


## Set up Pi Camera

[Via blog](https://larrylisky.com/2016/11/24/enabling-raspberry-pi-camera-v2-under-ubuntu-mate/)

    sudo apt-get install raspi-config rpi-update

## TinyYOLO

### Download Model

  * [Homepage](https://pjreddie.com/darknet/yolo/)
  * [TinyYOLO Keras-friendly H5 File](https://drive.google.com/open?id=1zm4diNjmf1-MOwFTQ8QhPrBSpQHJ1JM5)

## Future Optimizations

  * move to openSUSE
  * overclock the Raspberry Pi
  * ???


# Code

## Deep Learning Utilities

This `DLUtils` package contains utilities we'll want to use in:

  * scripts
  * webservices
  * iPython notebooks

From the directory containing `setup.py`, install in your virtualenv thusly:

    pip install --editable .

Now the packages listed in `setup.py` are installed in the current `pip` environment's library and available to all of the things we want to use them in listed above!

## Web Services

  * [REST Server](webservices/rest_server/README.md)
  * [Web GUI Client for REST Service](webservices/webui/README.md)

## Scripts

Scalable scripts.


## Notebooks

Exploratory & development iPython notebooks.