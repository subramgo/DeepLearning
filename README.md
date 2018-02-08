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

  1. Use 64-bit Rasbian
  2. [Tensorflow install](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
  3. `sudo pip3 install keras pyyaml`
    * if keras doesn't install correctly, use `apt`
  4. `sudo apt install python3-scipy` 
    * for some reason scipy wouldn't install via pip
  5. H5PY
    * Install `libhdf5-dev` first
        * `sudo apt install libhdf5-dev -y`
    * Takes a long time to compile and install
        * try verbose mode `sudo pip3 install h5py -vvv`
  6. OpenCV
    * [????](http://cyaninfinite.com/tutorials/installing-opencv-in-ubuntu-for-python-3/)
    * [????](https://raspberrypi.stackexchange.com/questions/69169/how-to-install-opencv-on-raspberry-pi-3-in-raspbian-jessie)


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

  * 64-bit compatibility
  * arm7l targetted compilation
  * overclock the Raspberry Pi
  * check battery usage on realtime YOLO


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