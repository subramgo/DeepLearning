"""
    Image Data Streaming Sources
"""

import cv2 as _cv2

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
        """ Together with __exit__, allows support for with- clauses. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        """ Together with __enter__, allows support for with- clauses. """
        pass

    def frame_generator(self):
        pass

    def preview_stream(self):
        """ Display stream until "q" key is pressed """
        pass

###################################################
###############   Picamera Stream   ###############
###################################################
from picamera.array import PiRGBArray as _PiRGBArray
from picamera import PiCamera as _PiCamera

class PiCam(Stream):
    """ Wrapper for Picamera stream source """

    def __init__(self,*args,**kwargs):
        print("opening stream to PiCamera")
        self._camera = _PiCamera(framerate=3, resolution = (640,480) )
        self._camera.rotation=180

    def __enter__(self,*args,**kwargs):
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        print("closing stream")
        self._camera.close()

    def close(self):
        self.__exit__()

    def frame_generator(self):
        rawCapture = _PiRGBArray(self._camera)
        self._camera.capture(rawCapture,format="bgr")
        frame = rawCapture.array
        while frame is not None:
            yield frame
            rawCapture.truncate(0)
            self._camera.capture(rawCapture,format="bgr")
            frame = rawCapture.array

    def preview_stream(self):
        """ Display stream in an OpenCV window until "q" key is pressed """
        for frame in self.frame_generator():
            if frame is not None:
                _cv2.imshow('Video', frame)
            else:
                break
            if _cv2.waitKey(24) & 0xFF == ord('q'):
                break
        _cv2.destroyAllWindows()


###################################################
##############   Flat File Stream   ###############
###################################################
""" Access a directory of image files, and provide
    them in a loop to simulate a stream. """
    #TODO

###################################################
###############    OpenCV Stream    ###############
###################################################
class OpenCVStream(Stream):
    """ Wrapper for OpenCV stream sources to support with-clauses and other conveniences """

    def __init__(self,*args,**kwargs):
        print("opening stream to {}".format(args[0]))
        self._stream = _cv2.VideoCapture(*args,**kwargs)

    def __enter__(self,*args,**kwargs):
        """ Together with __exit__, allows support for with- clauses. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        """ Together with __enter__, allows support for with- clauses. """
        print("closing stream")
        self._stream.release()
        _cv2.destroyAllWindows()

    def frame_generator(self):
        still_open,frame = self._stream.read()
        while still_open:
            yield frame
            still_open,frame = self._stream.read()

    def preview_stream(self):
        """ Display stream in an OpenCV window until "q" key is pressed """
        while(1):
            ret,frame = self._stream.read()
            if ret == False:
                print("Couldn't open stream")
                break
            else:
                _cv2.imshow('VIDEO', frame)
            key = _cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

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

if __name__ == '__main__':
    usb_stream()
    #cafe_stream()
