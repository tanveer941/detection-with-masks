
# import numpy as np
import os
# from PIL import Image
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import sys
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from utils import visualization_utils as vis_util
from collections import defaultdict
from datetime import datetime
from _datetime import datetime
import time
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from utils import ops as utils_ops


st_date = datetime.now()

if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)

app = Flask(__name__)
api = Api(app)


#----------------Trained model--------------------------------------
# FROZEN_MODEL = 'frozen_detection_model.pb'
FROZEN_MODEL = 'frozen_inference_graph_mask.pb'
CATEGORY_INDEX_FILE = 'category_index.json'


#----------------Training model--------------------------------------
# FROZEN_MODEL = 'frozen_inference_graph_5111.pb'
# FROZEN_MODEL = 'frozen_inference_graph_8151.pb'
# FROZEN_MODEL = 'frozen_inference_graph_8725.pb'
# CATEGORY_INDEX_FILE = 'category_index_rc.json'



IMAGE_SIZE = (12, 8)
# IMAGE_PATH = os.getcwd() + '/image3.jpg'
# IMAGE_PATH = 'image3.jpg'
NUMBER_OF_DETECTIONS = 20

detection_graph = tf.Graph()
with detection_graph.as_default():
  print("loading model......")
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(FROZEN_MODEL, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def read_json(file_path):
    with open(file_path) as outfile:
        output_data = json.load(outfile)
        return output_data

category_index = read_json(CATEGORY_INDEX_FILE)

# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#         print("detection begins......")
#         # Get handles to input and output tensors
#         ops = tf.get_default_graph().get_operations()
#         all_tensor_names = {output.name for op in ops for output in
#                             op.outputs}
#         tensor_dict = {}
#         for key in [
#             'num_detections', 'detection_boxes', 'detection_scores',
#             'detection_classes', 'detection_masks'
#         ]:
#             tensor_name = key + ':0'
#             if tensor_name in all_tensor_names:
#                 tensor_dict[
#                     key] = tf.get_default_graph().get_tensor_by_name(
#                     tensor_name)
        # if 'detection_masks' in tensor_dict:
        #     # The following processing is only for single image
        #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],
        #                                  [0])
        #     detection_masks = tf.squeeze(tensor_dict['detection_masks'],
        #                                  [0])
        #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        #     real_num_detection = tf.cast(
        #         tensor_dict['num_detections'][0], tf.int32)
        #     detection_boxes = tf.slice(detection_boxes, [0, 0],
        #                                [real_num_detection, -1])
        #     detection_masks = tf.slice(detection_masks, [0, 0, 0],
        #                                [real_num_detection, -1, -1])
        #     print(">>>", image_np_expanded.shape, image_np_expanded.shape[0],
        #           image_np_expanded.shape[1])
        #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        #         detection_masks, detection_boxes, image_np_expanded.shape[0],
        #         image_np_expanded.shape[1])
        #     detection_masks_reframed = tf.cast(
        #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        #     # Follow the convention by adding back the batch dimension
        #     tensor_dict['detection_masks'] = tf.expand_dims(
        #         detection_masks_reframed, 0)
        # image_tensor = tf.get_default_graph().get_tensor_by_name(
        #     'image_tensor:0')

duration = datetime.now() - st_date
print("Duration to load model...", duration)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed

# @app.route('/detect')
class ObjectDetect(Resource):


    # @timeit
    def get(self):
        # print("------------->>>>", IMAGE_PATH)
        parser = reqparse.RequestParser()
        parser.add_argument('image_path', type=str)
        args = parser.parse_args()
        image_path = args['image_path']
        image = Image.open(image_path)
        # print("image :: ", image, type(image), dir(image))
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # print("image_np >> ", image_np, type(image_np), image_np.shape)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        image_np_expanded = image_np
        # print("image_np_expanded >> ", image_np_expanded,
        #       type(image_np_expanded), image_np_expanded.shape)

        st_time = datetime.now()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                print("detection begins......")
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in
                                    op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[
                            key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],
                                                 [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'],
                                                 [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0],
                                               [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                               [real_num_detection, -1, -1])
                    # print(">>>", image_np_expanded.shape, image_np_expanded.shape[0], image_np_expanded.shape[1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image_np_expanded.shape[0],
                        image_np_expanded.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')

                et_time = datetime.now()
                # duration =
                print("session loaded in ... ", et_time - st_time)
                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={
                                           image_tensor: np.expand_dims(image_np_expanded,
                                                                        0)})
                ed_time = datetime.now()
                duration = ed_time - st_time
                print("detection done in ... ", duration)
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][
                    0]
                output_dict['detection_scores'] = \
                output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = \
                    output_dict['detection_masks'][0]
                # print("output_dict :: ", output_dict)
                image_np_op = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # plt.figure(figsize=IMAGE_SIZE)
                # plt.imshow(image_np_op)

                cv2.imwrite('color_img.jpg', image_np_op)
                cv2.imshow('Color image', image_np_op)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return output_dict


api.add_resource(ObjectDetect, '/detect')

if __name__ == '__main__':
    # category_index = read_json(CATEGORY_INDEX_FILE)
    # print("-->> ", category_index, type(category_index))
    app.run()

# C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile obj_detect_lite.py --add-data frozen_detection_model.pb;. --add-data category_index.json;.

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\imag_test2\MFC4xxLongImageRight_1470898323225222.jpeg

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\Images_Line\LongImage_07.jpeg

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\Images\right_1486411298204826.jpeg

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\Images_outside\MFC4xxLongImageRight_1470898245640948.jpeg

# Traffic light
# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\LD\Images_outgoing\MFC4xxLongImageRight_1486411409304678.jpeg



































#C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile obj_detect_lite.py --add-data frozen_detection_model.pb;. --add-data category_index.json;.


# http://127.0.0.1:5000/detect?image_path=D:\TDSM\MKS_integration_before\TDSM_auth\TF\TF_prac\TF_V1_May12\TF_run\object_detection_TF\obj_detect_make_exe\image3.jpg

# http://127.0.0.1:5000/detect?image_path=D:\Work\2018\code\Tensorflow_code\object_detection\tfl_masking_detect\mask_lite\MFC4xxLongImageRight_1470898348001556.jpeg
#