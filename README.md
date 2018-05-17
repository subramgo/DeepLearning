# Tasks


## Future Optimizations

  * optimize tensorflow build for our platform
  * remove opencv dependencies
  * optimize `cmake` build of opencv for our platform
    * [cmake config viewer](https://stackoverflow.com/a/42945360/1949791)
    * [a head-start on opencv optimization](http://amritamaz.net/blog/opencv-config)
  * remove HDF5 dependencies
  * 64-bit compatibility everywhere?
  * arm7l targetted compilation
  * overclock the Raspberry Pi
  * check battery usage on realtime YOLO

# Raspberry Pi

### SD Card Cloning

Locate the SD card using `diskutil list`.

    sudo dd if=/dev/rdisk2 | gzip -c > ~/Desktop/raspberrypi.dmg.zip

  * `dd` for byte copying
  * `rdisk2` instead of `disk2` for faster access on Mac
  * `gzip` to compress output, try to quickly remove empty space from image

### OS Prep

  1. Use 64-bit Raspbian. 
  2. Python 3.4.2 makes the other installs easier/possible.
  3. `sudo apt purge wolfram-engine && sudo apt autoremove`
  3. `sudo apt update && sudo apt upgrade && sudo rpi-update && sudo reboot`
  4. `sudo apt install build-essential screen vim git python3-dev libqt4-dev cmake pkg-config libatlas-base-dev`
  5. `sudo apt update && sudo apt upgrade -y`
  6. `sudo pip3 install numpy`

### Important Packages

Don't run `pip install -r requirements.txt` on a Pi as it will choke on the large packages (Tensorflow, Keras, H5PY, OpenCV).

#### H5PY

    sudo apt install libhdf5-dev python3-h5py python3-scipy -y

#### [Tensorflow](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md)

Installing a `...cp34...` by renaming to `cp35` will result in binary incompatability and bus errors.

##### Build from Source

  1. `sudo apt-get install python3-pip python3-dev`
  2. [Increase Raspbian swap space](https://www.bitpi.co/2015/02/11/how-to-change-raspberry-pis-swapfile-size-on-rasbian/)
      1. /etc/dphys-swapfile -> `CONF_SWAPFILE=1024`
      2. `sudo /etc/init.d/dphys-swapfile stop; sudo /etc/init.d/dphys-swapfile start`
  3. build `bazel` from source 
  4. [build `tensorflow` using `bazel`](https://www.tensorflow.org/install/install_sources)
    * `bazel build -c opt --copt="-mfpu=neon-vfpv4" --copt="-funsafe-math-optimizations" --copt="-ftree-vectorize" --copt="-fomit-frame-pointer" --local_resources 1024,1.0,1.0 --verbose_failures tensorflow/tools/pip_package:build_pip_package --config=monolithic`
  5. `sudo pip3 install mock`


On Raspbian 9.3 Stretch, python 3.5, and Bazel 0.10.0.

The output wheel is `tensorflow-1.6.0rc1-cp35-cp35m-linux_armv7l.whl`.

`pip3 install tensorflow-1.6.0rc1-cp35-cp35m-linux_armv7l.whl` 

#### Keras & OpenCV

    sudo pip3 install keras pyyaml opencv-python

  * if keras doesn't install correctly, use `apt`


## Set up Pi Camera

  * [Quick Start](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera)
  * [Basic Docs](https://www.raspberrypi.org/documentation/usage/camera/python/README.md)
  * [Full Docs](http://picamera.readthedocs.io/en/release-1.13/recipes2.html)

  5. https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/ connect picamera to OpenCV


## TinyYOLO

### Download Model

  * [Homepage](https://pjreddie.com/darknet/yolo/)
  * [TinyYOLO Keras-friendly H5 File](https://drive.google.com/open?id=1zm4diNjmf1-MOwFTQ8QhPrBSpQHJ1JM5)

# Code

## Deep Learning Utilities

This `DLUtils` package contains utilities we'll want to use in:

  * scripts
  * webservices
  * iPython notebooks

From the directory containing `setup.py`, install in your virtualenv thusly:

    pip install --editable .

Now the packages listed in `setup.py` are installed in the current `pip` environment's library and available to all of the things we want to use them in listed above!

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

# Projects

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

