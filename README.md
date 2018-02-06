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

  * [Ubuntu Desktop](https://www.ubuntu.com/download)
  * [Ubuntu Mate](https://ubuntu-mate.org/raspberry-pi/)
    - current Ubuntu desktop with `apt` support
    - preconfigured for `armhf` base packages

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

### OpenCV

Install OpenCV and the `requirements.txt`.

  1. Use PIP
    * `pip3 install opencv-python`
  2. Use APT
    * `sudo apt install python-opencv`
  3. Build from source
    * use https://opencv.org

### Tensorflow

  * Build it from source also
  * Use [this build](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
    * change 'cp34-cp34m' to read 'cp35-cp35m'


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