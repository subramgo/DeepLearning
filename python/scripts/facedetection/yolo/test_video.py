import cv2
import argparse
import colorsys
import imghdr
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from DLUtils.cellar.yolo.yolokeras import yolo_eval, yolo_head
from DLUtils import datafeed
import time
import scipy.misc

anchors = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])

class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
yolo_model = load_model("../models/yolo/tiny_yolo.h5")
num_classes = len(class_names)
num_anchors = len(anchors)

# TODO: Assumes dim ordering is channel last
model_output_channels = yolo_model.layers[-1].output_shape[-1]
assert model_output_channels == num_anchors * (num_classes + 5), \
    'Mismatch between model and given anchor and class sizes. ' \
    'Specify matching anchors and classes with --anchors_path and ' \
    '--classes_path flags.'
print('Tiny Yolo model, anchors, and classes loaded.')

# Check if model is fully convolutional, assuming channel last order.
model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

score_threshold = 0.3
iou_threshold = 0.5

# Generate output tensor targets for filtered bounding boxes.
# TODO: Wrap these backend operations with Keras layers.
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=score_threshold,
    iou_threshold=iou_threshold)

def _main(image):
    verbose=True

    #print(image.shape)
    #sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    # Verify model, anchors, and classes are compatible


    print(image.size)
    print(model_image_size)
    #image = np.asarray(image)
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            #resized_image = image.resize((416, 416), Image.BICUBIC)
            #resized_image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
            image_data = np.array(resized_image, dtype='float32')
    if verbose: print("start of main()")

    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
        #resized_image = cv2.resize(image, (416, 416), interpolation = cv2.INTER_CUBIC)
        image_data = np.array(resized_image, dtype='float32')
        if verbose: print("resized image")

    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    if verbose: print("added batch dimension")
    feed_dict={
        yolo_model.input: image_data,
        input_image_shape: [image.size[1], image.size[0]],
        K.learning_phase(): 0}

    if verbose: print("made feed dict")

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict=feed_dict)

    font = ImageFont.truetype(
                              font='/usr/share/fonts/truetype/lato/Lato-Medium.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    #font = 1

    if verbose: print("start prediction")

    for i, c in reversed(list(enumerate(out_classes))):
        c = int(c)
        predicted_class = class_names[int(c)]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(-1)
    video_capture.set(4,640)
    video_capture.set(5,480)
    video_capture.set(6,15)
    while True:
        ret, frame = video_capture.read()
        print(ret)
        if ret:
            #cv2.imshow('Video', frame)
            if frame is not None:

                cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                #frame = scipy.misc.toimage(frame)
                frame = _main(frame)
                cv2.imshow('Video', np.asarray(frame))
                #time.sleep(1)
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    sess.close()

"""
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    import cv2

    camera = PiCamera()
    camera.resolution = (640,480)
    rawCapture = PiRGBArray(camera)

    time.sleep(0.1)

    camera.capture(rawCapture,format="bgr")
    frame = rawCapture.array
    cv2.imshow('Video',frame)
    rawCapture.truncate(0)


    for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
        time.sleep(1)
        #camera.capture(rawCapture,format="bgr")
        #frame = rawCapture.array

        ## Need a numpy.ndarray, shape of (height,width,dims)
        #cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame.array)

        frame = _main(frame)
        cv2.imshow('Video', np.asarray(frame))
        key = cv2.waitKey(1) & 0xFF

        rawCapture.truncate(0)

        if key == ord("q"):
            break

        # No loop now
        break


    capture.release()
>>>>>>> 6060ad8124dad80df86b143145000e6567897d1c
    cv2.destroyAllWindows()
    sess.close()

    """
