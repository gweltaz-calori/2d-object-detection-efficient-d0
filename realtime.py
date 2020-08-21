import numpy as np
import tensorflow as tf
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt

import os
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

pipeline_config = "./saved_model/pipeline.config"
model_dir = './saved_model/ckpt-15'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join('./saved_model/ckpt-15'))

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)

def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)

cap = cv2.VideoCapture('video2.mp4')
frame_rate = 60
prev = 0
square = 512

while(cap.isOpened()):
    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    resize = resize2SquareKeepingAspectRation(frame, square, cv2.INTER_AREA)

    if ret:
      prev = time.time()
      input_tensor = tf.convert_to_tensor(np.expand_dims(resize, 0), dtype=tf.float32)
      detections, predictions_dict, shapes = detect_fn(input_tensor)
      image_np_with_detections = resize.copy()

      boxes = detections['detection_boxes'][0].numpy()
      classes = detections['detection_classes'][0].numpy() + 1
      scores = detections['detection_scores'][0].numpy()

      maxScore = 0
      maxIndex = 0 
      for i in range(boxes.shape[0]):
          if(scores[i] > maxScore):
            maxScore = scores[i]
            maxIndex = i
      box = tuple(boxes[maxIndex].tolist())
      ymin, xmin, ymax, xmax = box
      (left, right, top, bottom) = (xmin * square, xmax * square,
                    ymin * square, ymax * square)
      cv2.rectangle(image_np_with_detections,(int(left),int(top)),(int(right),int(bottom)),(255,0,0),2)
      """ if maxScore > .99:
        box = tuple(boxes[maxIndex].tolist())
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * square, xmax * square,
                      ymin * square, ymax * square)
        cv2.rectangle(image_np_with_detections,(int(left),int(top)),(int(right),int(bottom)),(255,0,0),2) """
      cv2.imshow('image_np_with_detections',image_np_with_detections)      

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()