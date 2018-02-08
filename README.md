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

### OS Prep

  1. Use 64-bit Raspbian. 
  2. Python 3.4.2 makes the other installs easier/possible.
  3. `sudo apt update && sudo apt upgrade && sudo rpi-update && sudo reboot`
  4. `sudo apt install build-essential screen vim git python3-dev cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libatlas-base-dev gfortran`
  5. `sudo apt update && sudo apt upgrade -y`
  6. `sudo pip3 install numpy`

### Important Packages

  3. [Tensorflow install](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
  4. `sudo pip3 install keras pyyaml`
      * if keras doesn't install correctly, use `apt`
  5. `sudo apt install python3-scipy` 
      * for some reason scipy wouldn't install via pip
  6. H5PY
      * Install `libhdf5-dev` first
          * `sudo apt install libhdf5-dev -y`
      * Takes a long time to compile and install
          * try verbose mode `sudo pip3 install h5py -vvv`
  7. OpenCV
      * Use PIP
          * `pip3 install opencv-python`
      * [Build from source](https://opencv.org)
          1. `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..`
          2. `make -j4`
          3. `sudo make install`
          4. `sudo ldconfig`
      * [source1](https://raspberrypi.stackexchange.com/questions/69169/how-to-install-opencv-on-raspberry-pi-3-in-raspbian-jessie)
      * [source2](http://cyaninfinite.com/tutorials/installing-opencv-in-ubuntu-for-python-3/)


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

  * remove HDF5 dependencies
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