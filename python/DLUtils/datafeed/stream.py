"""
    Image Data Streaming Sources
"""

import cv2 as _cv2


###################################################
###############    OpenCV Stream    ###############
###################################################
class CVStream(_cv2.VideoCapture):
    """ Wrapper for OpenCV stream sources to support with-clauses and other conveniences """

    def init(self,kwargs):
        super(kwargs)

    def __enter__(self):
        """ Together with __exit__, allows support for with- clauses. """
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        """ Together with __enter__, allows support for with- clauses. """
        self.release()
        _cv2.destroyAllWindows()

    def show_stream(self):
        """ Display stream in an OpenCV window until "q" key is pressed """
        while(1):
            ret,frame = self.read()
            if ret == False:
                print("Ret is false")
            else:
                _cv2.imshow('VIDEO', frame)
                key = _cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

def cafe_uri():
    return "rtsp://10.38.5.145/ufirststream/"

def cafe_stream():
    """ stream object """
    with CVStream(cafe_uri()) as cap:
        cap.show_stream()

if __name__ == '__main__':
    cafe_stream()