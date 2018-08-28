
# import numpy as np
import os
# from PIL import Image
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import sys
import tensorflow as tf
import json
import collections
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw

from datetime import datetime
from _datetime import datetime
import time
import cv2
import numpy as np
from PIL import Image

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

    def reframe_box_masks_to_image_masks(self, box_masks, boxes, image_height,
                                         image_width):
        """Transforms the box masks back to full image masks.

        Embeds masks in bounding boxes of larger masks whose shapes correspond to
        image shape.

        Args:
          box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
          boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
                 corners. Row i contains [ymin, xmin, ymax, xmax] of the box
                 corresponding to mask i. Note that the box corners are in
                 normalized coordinates.
          image_height: Image height. The output mask will have the same height as
                        the image height.
          image_width: Image width. The output mask will have the same width as the
                       image width.

        Returns:
          A tf.float32 tensor of size [num_masks, image_height, image_width].
        """

        # TODO: Make this a public function.
        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        image_masks = tf.image.crop_and_resize(image=box_masks,
                                               boxes=reverse_boxes,
                                               box_ind=tf.range(num_boxes),
                                               crop_size=[image_height,
                                                          image_width],
                                               extrapolation_value=0.0)
        return tf.squeeze(image_masks, axis=3)

    def get_detection_details(self, image,
                              boxes,
                              classes,
                              scores,
                              category_index,
                              instance_masks=None,
                              use_normalized_coordinates=False,
                              max_boxes_to_draw=20,
                              min_score_thresh=.5,
                              line_thickness=4):

        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_instance_boundaries_map = {}
        box_to_keypoints_map = collections.defaultdict(list)
        obj_details_lst = []

        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                classes_keys_str_lst = [str(ech_class) for ech_class in
                                        category_index.keys()]
                # print(">>?? ", category_index.keys())
                # print("{{{{", classes_keys_str_lst)
                if str(classes[i]) in classes_keys_str_lst:
                    class_name = category_index[str(classes[i])]['name']
                    score = int(100 * scores[i])
                else:
                    class_name = 'N/A'
                # print("class_name :: ", class_name, score)
                ymin, xmin, ymax, xmax = box
                image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                im_width, im_height = image_pil.size
                if use_normalized_coordinates:
                    (left, right, top, bottom) = (
                    xmin * im_width, xmax * im_width,
                    ymin * im_height, ymax * im_height)
                else:
                    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                # print("score :> ", score)
                # print("co-ordinates :: ", (left, right, top, bottom))
                # Get the mask for each object detected
                mask = instance_masks[i]
                # print("mask :: ", mask)

                rgb = ImageColor.getrgb('red')
                pil_image = Image.fromarray(image)
                # tz
                solid_color = np.expand_dims(
                    np.ones_like(mask), axis=2) * np.reshape(list(rgb),
                                                             [1, 1, 3])
                pil_solid_color = Image.fromarray(
                    np.uint8(solid_color)).convert(
                    'RGBA')
                pil_mask = Image.fromarray(
                    np.uint8(255.0 * 0.4 * mask)).convert(
                    'L')
                pil_image = Image.composite(pil_solid_color, pil_image,
                                            pil_mask)
                mask_layer = np.array(pil_mask.convert('RGB'))
                mask_layer_string = mask_layer.tostring()
                np.copyto(image, np.array(pil_image.convert('RGB')))
                # return np.array(pil_mask.convert('RGB'))
                # Draw bounding box on the image
                image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                # print(">> ", type(image_pil))
                draw = ImageDraw.Draw(image_pil)
                # im_width, im_height = image.size
                draw.line([(left, top), (left, bottom), (right, bottom),
                           (right, top), (left, top)], width=line_thickness,
                          fill='FloralWhite')
                np.copyto(image, np.array(image_pil))

                # obj_details_lst.append(
                #     {'class': class_name, 'score': score, 'xmin': left,
                #      'xmax': right, 'ymin': top, 'ymax': bottom,
                #      'mask': mask_layer_string})

                obj_details_lst.append(
                    {'class': class_name, 'score': score, 'xmin': left,
                     'xmax': right, 'ymin': top, 'ymax': bottom
                     })
                #
                print("\n")

        # print("obj_details_lst :: ", obj_details_lst)
        # return obj_details_lst
        # return np.array(pil_mask.convert('RGB'))
        return image, obj_details_lst

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
                    detection_masks_reframed = self.reframe_box_masks_to_image_masks(
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

                image, detection_details = self.get_detection_details(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=2)

                cv2.imwrite('color_img.jpg', image)
                cv2.imshow('Color image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # return jsonify("Detection done!!!")
                # del ech_detect_dict['mask']
                # output_list = []
                # for ech_detect_dict in detection_details.deepcopy():
                #     for k, v in ech_detect_dict.items():
                #         if k == 'mask':
                #             del ech_detect_dict['mask']
                #         output_list.append(ech_detect_dict)
                return detection_details


api.add_resource(ObjectDetect, '/detect')

if __name__ == '__main__':
    # category_index = read_json(CATEGORY_INDEX_FILE)
    # print("-->> ", category_index, type(category_index))
    app.run()

# C:\Users\uidr8549\Envs\python3_5_tf\Scripts\pyinstaller --onefile lt5_tf_mask_detect.py --add-data frozen_inference_graph_mask.pb;. --add-data category_index.json;.

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\imag_test2\MFC4xxLongImageRight_1470898323225222.jpeg

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\Images_Line\LongImage_07.jpeg

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\Images\right_1486411298204826.jpeg

# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\Images_outside\MFC4xxLongImageRight_1470898245640948.jpeg

# Traffic light
# http://127.0.0.1:5000/detect?image_path=D:\TDSM\LT5G\Ticket_folders\LD\Images_outgoing\MFC4xxLongImageRight_1486411409304678.jpeg



































#C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile obj_detect_lite.py --add-data frozen_detection_model.pb;. --add-data category_index.json;.


# http://127.0.0.1:5000/detect?image_path=D:\TDSM\MKS_integration_before\TDSM_auth\TF\TF_prac\TF_V1_May12\TF_run\object_detection_TF\obj_detect_make_exe\image3.jpg
#