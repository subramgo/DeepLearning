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

# Raspberry Pi

### OS Prep

  1. Use 64-bit Raspbian. 
  2. Python 3.4.2 makes the other installs easier/possible.
  3. `sudo apt purge wolfram-engine && sudo apt autoremove`
  3. `sudo apt update && sudo apt upgrade && sudo rpi-update && sudo reboot`
  4. `sudo apt install build-essential screen vim git python3-dev cmake pkg-config libatlas-base-dev`
  5. `sudo apt update && sudo apt upgrade -y`
  6. `sudo pip3 install numpy`

### Important Packages

Don't run `pip install -r requirements.txt` on a Pi as it will choke on the large packages (Tensorflow, Keras, H5PY, OpenCV).

#### H5PY

    sudo apt install libhdf5-dev python3-h5py python3-scipy -y

#### [Tensorflow](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md)

##### Install Nightly Build

  1. `wget http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/39/artifact/output-artifacts/tensorflow-1.4.0-cp34-none-any.whl`
  2. `pip3 install tensorflow-...`

##### Build from Source

  1. `sudo apt-get install python3-pip python3-dev`
  2. [Increase Raspbian swap space](https://www.bitpi.co/2015/02/11/how-to-change-raspberry-pis-swapfile-size-on-rasbian/)
    1. /etc/dphys-swapfile -> `CONF_SWAPFILE=1024`
    2. `sudo /etc/init.d/dphys-swapfile stop; sudo /etc/init.d/dphys-swapfile start`
  3. build `bazel` from source 
  4. build `tensorflow` using `bazel`
  5. `sudo pip3 install mock`

#### Keras

    sudo pip3 install keras pyyaml
  * if keras doesn't install correctly, use `apt`

#### [OpenCV](https://opencv.org) [via SE](https://raspberrypi.stackexchange.com/questions/69169/how-to-install-opencv-on-raspberry-pi-3-in-raspbian-jessie)
  1. `sudo apt install libgtk2.0-dev and pkg-config -y` GTK2 provides critical functions
  2. download [opencv](https://github.com/opencv) and [opencv_contrib](https://github.com/opencv/opencv_contrib)
  3. check out same release version of each
  4. build opencv
      1. `cd opencv && mkdir build && cd build`
      2. `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/workspace/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..`
      3. `make -j3`
          * if it appears to freeze or stall, don't panic. be patient. When patience runs out, hit CTRL-C to cancel and resume build with a `make`
      4. `sudo make install`
      5. `sudo ldconfig`
  5. https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/ connect picamera to OpenCV

  sudo apt install imagemagick

  pip3 install wand

### PyQT

  * Try installing by `apt`
    * `sudo apt install build-essential python3-dev libqt4-dev`



## Set up Pi Camera

  * [Quick Start](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera)
  * [Basic Docs](https://www.raspberrypi.org/documentation/usage/camera/python/README.md)
  * [Full Docs](http://picamera.readthedocs.io/en/release-1.13/recipes2.html)

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

