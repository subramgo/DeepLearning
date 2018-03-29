"""
    Image Data Streaming Sources
"""

class Stream:
    """
        Simple interface for stream sources, such as USB-camera, PiCamera, and RTSP servers.
        Shows these features:
            * context manager for opening the stream
            * open/close methods for non-context-management
            * frame generator
            * `preview_stream()` to demo in GTK
    """

    def __init__(self,*args,**kwargs):
        pass

    def __enter__(self,*args,**kwargs):
        """ Returns the object which later will have __exit__ called.
            This relationship creates a context manager. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        """ Together with __enter__, allows support for with- clauses. """
        pass

    def frame_generator(self):
        pass

    def preview_stream(self):
        """ Display stream in an OpenCV window until "q" key is pressed """
        import cv2 as _cv2
        for frame in self.frame_generator():
            if frame is not None:
                _cv2.imshow('Video', frame)
            else:
                break
            #if _cv2.waitKey(24) & 0xFF == ord('q'):
            #    break
            key = _cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        _cv2.destroyAllWindows()

###################################################
###############   Picamera Stream   ###############
###################################################

class PiCam(Stream):
    """ Wrapper for Picamera stream source """

    def __init__(self,*args,**kwargs):
        from picamera.array import PiRGBArray as _PiRGBArray
        from picamera import PiCamera as _PiCamera
        self._PiRGBArray = _PiRGBArray
        self._PiCamera = _PiCamera

        print("opening stream to PiCamera")
        self._camera = self._PiCamera(framerate=3, resolution = (640,480) )
        self._camera.rotation=180

    def __enter__(self,*args,**kwargs):
        """ Returns the object which later will have __exit__ called.
            This relationship creates a context manager. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        print("closing stream")
        self._camera.close()

    def close(self):
        self.__exit__()

    def frame_generator(self):
        rawCapture = self._PiRGBArray(self._camera)
        self._camera.capture(rawCapture,format="bgr")
        frame = rawCapture.array
        while frame is not None:
            yield frame
            rawCapture.truncate(0)
            self._camera.capture(rawCapture,format="bgr")
            frame = rawCapture.array



###################################################
##############   Flat File Stream   ###############
###################################################
class FileStream(Stream):
    """ Access a directory of image files, and provide
        them in a loop to simulate a stream. """
    def __init__(self,path,loop=True):
        """ Grab list of files and sort by numbers in the file names. """
        import os
        import re

        self._loop = loop
        self._path = path

        _files = os.listdir(path)
        print("Found {} files in {}.".format(len(_files),path))

        _files = list(zip(_files, [int(re.findall('\d+',name)[0]) for name in _files]))
        _files.sort(key =  lambda x: x[1])
        self._files = _files

    def __enter__(self,*args,**kwargs):
        """ Returns the object which later will have __exit__ called.
            This relationship creates a context manager. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        """ Together with __enter__, allows support for with- clauses. """
        self._stream.release()
        self._cv2.destroyAllWindows()

    def frame_generator(self):
        from PIL import Image
        import scipy.misc
        import os
        import numpy as np
        _files = self._files
        while True:
            for image in _files:
                try:
                    yield np.asarray(scipy.misc.toimage(Image.open(os.path.join(self._path,_files.pop()[0]))))
                except IndexError:
                    pass
            print("used all files")
            if not self._loop:
                break
        self.__exit__()


###################################################
#############   HDF5 File Stream    ###############
###################################################
""" Access a video, and provide frames stream. """
    #TODO

###################################################
###############    OpenCV Stream    ###############
###################################################
class OpenCVStream(Stream):
    """ Wrapper for OpenCV stream sources to support with-clauses and other conveniences """

    def __init__(self,*args,**kwargs):
        import cv2 as _cv2
        self._cv2 = _cv2

        print("opening stream to {}".format(args[0]))
        self._stream = _cv2.VideoCapture(*args,**kwargs)

    def __enter__(self,*args,**kwargs):
        """ Returns the object which later will have __exit__ called.
            This relationship creates a context manager. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        """ Together with __enter__, allows support for with- clauses. """
        print("closing stream")
        self._stream.release()
        self._cv2.destroyAllWindows()

    def frame_generator(self):
        still_open,frame = self._stream.read()
        while still_open:
            yield frame
            still_open,frame = self._stream.read()

###################################################
###############       Examples      ###############
###################################################
def _cafe_uri():
    return "rtsp://10.38.5.145/ufirststream/"

def demo_picam():
    with PiCam() as src:
        src.preview_stream()

def demo_usb():
    with OpenCVStream(-1) as cap:
        cap.preview_stream()

def demo_rtsp():
    """ stream object """
    with OpenCVStream(_cafe_uri()) as cap:
        cap.preview_stream()

def demo_filestream():
    path = "/data/Walter/frames"
    with FileStream(path) as src:
        src.preview_stream()

if __name__ == '__main__':
    demo_usb()
    #demo_rtsp()
