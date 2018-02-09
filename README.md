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
  7. [OpenCV](https://opencv.org) [via SE](https://raspberrypi.stackexchange.com/questions/69169/how-to-install-opencv-on-raspberry-pi-3-in-raspbian-jessie)
      1. download [opencv](https://github.com/opencv) and [opencv_contrib](https://github.com/opencv/opencv_contrib)
      2. check out same release version of each
      3. build opencv
          1. `cd opencv && mkdir build && cd build`
          2. `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/workspace/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..`
          3. `make -j4`
          4. `sudo make install`
          5. `sudo ldconfig`


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

# Raspberry Pi Troubleshooting

## Custom Pi Builds

### SDHC Preparation

Use a high-speed-rated SD card if rapid writes are needed, such as "10" or greater as used in GoPro.
Easiest & reliable way to install an OS image using `etcher`.

  * [openSUSE Linux](https://en.opensuse.org/HCL:Raspberry_Pi3)
    * 64-bit runtime
  * [Ubuntu Mate](https://ubuntu-mate.org/raspberry-pi/)
    - current Ubuntu desktop with `apt` support
    - preconfigured for `armhf` base packages
    * only 32-bit runtime
  * [Ubuntu Desktop](https://www.ubuntu.com/download)


## Troubleshooting

Don't run `pip install -r requirements.txt` on a Pi as it will choke on the large packages Tensorflow, Keras, H5PY, OpenCV.

### PyQT

  * Try installing by `apt`
    * `sudo apt install build-essential python3-dev libqt4-dev`

### H5PY

  * apt install
    * `sudo apt install python3-h5py`
  * pip install
    1. Install `libhdf5-dev` first
      * `sudo apt install libhdf5-dev -y`
    2. Takes a long time to compile and install
      * try verbose mode `pip3 install h5py -vvv`

### OpenCV

  * Use PIP
    * `pip3 install opencv-python`
  * [Build from source](https://opencv.org) [via SE](https://raspberrypi.stackexchange.com/questions/69169/how-to-install-opencv-on-raspberry-pi-3-in-raspbian-jessie)
      1. download [opencv](https://github.com/opencv) and [opencv_contrib](https://github.com/opencv/opencv_contrib)
      2. check out same release version of each
      3. build opencv
          1. `cd opencv && mkdir build && cd build`
          2. `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/workspace/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..`
          3. `make -j4`
          4. `sudo make install`
          5. `sudo ldconfig`

### Tensorflow

  * [Easy install](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
  * [Build Tensorflow from source](https://www.tensorflow.org/install/install_sources)
  * [Cross-Compile](https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi/)
    1. get the [latest nightly build](http://ci.tensorflow.org/view/Nightly/job/nightly-pi-zero-python3/lastSuccessfulBuild/artifact/output-artifacts/)
    2. change 'cp34' to read 'cp35'
    3. `sudo pip3 install tensorflow-...`