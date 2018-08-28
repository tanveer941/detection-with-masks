
import ecal
import sys
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
import time
import tensorflow as tf
import json
from collections import defaultdict
import cv2
import PIL.ImageColor as ImageColor
import collections
import AlgoInterface_pb2
import imageservice_pb2
# tf version 1.1.0
BEXIT = True
ALGO_READINESS = True

# if getattr(sys, 'frozen', False):
#     os.chdir(sys._MEIPASS)

FROZEN_MODEL = 'frozen_inference_graph_mask.pb'
CATEGORY_INDEX_FILE = 'category_index.json'
IMAGE_SIZE = (12, 8)
NUMBER_OF_DETECTIONS = 20
TOPICS_JSON = 'topics.json'


class DetectionMasksROI(object):

    def __init__(self):
        # Initialize eCAL
        ecal.initialize(sys.argv, "object detection Pixel and ROI")
        # Read the JSON files
        with open(TOPICS_JSON) as data_file:
            self.json_data = json.load(data_file)
        with open(CATEGORY_INDEX_FILE) as data_file:
            self.category_index = json.load(data_file)
        # Load the detection model
        self.load_model_detection()
        # Define the callbacks for publisher subscriber
        self.initialize_subscr_topics()
        self.initialize_publsr_topics()
        # Inform the tool that model is loaded
        # self.inform_model_loaded()
        # The callbacks will redirect to the detection function and publish ROI
        self.define_subscr_callbacks()

    def load_model_detection(self):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            print("loading model......")
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                print("Model loaded.....")

    def initialize_subscr_topics(self):
        # Initialize all the subscriber topics
        self.lt5_img_subscr_obj = ecal.subscriber(self.json_data['image_request'])
        self.lt5_finl_subscr_obj = ecal.subscriber(self.json_data['algo_end_response'])

    def initialize_publsr_topics(self):
        # Initialize all the publisher topics
        self.lt5_img_publr_obj = ecal.publisher(self.json_data['image_response'])
        self.lt5_algo_publr_obj = ecal.publisher(self.json_data['algo_begin_response'])

    def filter_obj_class(self, obj_details_lst):
        with open('object_class_filter.json') as outfile:
            obj_class_json = json.load(outfile)
        # Get class names which are set to True
        # obj_cls_true_lst = []
        # for cls_name,cls_bool in obj_class_json.items():
        #     if cls_bool == 'True':
        #         obj_cls_true_lst.append(cls_name)
        filter_obj_class_lst = []
        for each_obj_dict in obj_details_lst:
            if each_obj_dict['class'] in obj_class_json:
                filter_obj_class_lst.append(each_obj_dict)
        return filter_obj_class_lst

    def get_detection_details(self, image,
                              boxes,
                              classes,
                              scores,
                              category_index,
                              instance_masks=None,
                              use_normalized_coordinates=False,
                              max_boxes_to_draw=20,
                              min_score_thresh=.5,
                              line_thickness=4,
                              visualize_mask=False):

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
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                  ymin * im_height, ymax * im_height)
                else:
                    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                # print("score :> ", score)
                # print("co-ordinates :: ", (left, right, top, bottom))
                # Get the mask for each object detected
                mask = instance_masks[i]
                # print("mask :: ", mask)

                rgb = ImageColor.getrgb('red')
                # print("rgb :: ", rgb)
                pil_image = Image.fromarray(image)
                # tz
                solid_color = np.expand_dims(
                    np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
                pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert(
                    'RGBA')
                pil_mask = Image.fromarray(np.uint8(255.0 * 0.4 * mask)).convert(
                    'L')
                pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
                mask_layer = np.array(pil_mask.convert('RGB'))
                # print("mask_layer :: ", mask_layer)
                # with open('img.txt', 'w+') as fhandle:
                #     fhandle.write(str(mask_layer))
                # mask_layer_string = mask_layer.tobytes()
                mask_layer_string = cv2.imencode('.png', mask_layer)[1].tostring()
                # print("mask_layer_string :: ", mask_layer_string)
                # cv2.imwrite('sample.png', mask_layer_string)
                # cv2.imwrite('color_img{}.jpg'.format(str(i)), mask_layer)
                # visualize_mask = True
                if visualize_mask:
                    # pass
                    # np.copyto(image, mask_layer)
                    np.copyto(image, pil_image)
                obj_details_lst.append({'class': class_name, 'score': score, 'xmin': left,
                                        'xmax': right, 'ymin': top, 'ymax': bottom, 'mask': mask_layer_string})
                #
                # print("class_name :: ", class_name)
                print("\n")
        fltrd_obj_lst = self.filter_obj_class(obj_details_lst)
        # print("fltrd_obj_lst :: ", fltrd_obj_lst)

        if not visualize_mask:
            return fltrd_obj_lst
        else:
            # Show masks
            # cv2.imwrite('color_img.jpg', mask_layer)
            # cv2.imshow('Color image', mask_layer)
            # Show image
            cv2.imwrite('color_img.jpg', image)
            cv2.imshow('Color image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

    def detect_obj(self, image_np_ary, width, height):
        # print("image_np_ary >> ", image_np_ary, type(image_np_ary), image_np_ary.shape)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = image_np_ary
        # image_np_expanded = np.expand_dims(image_np_ary, axis=0)

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
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
                        detection_masks, detection_boxes,
                        image_np_expanded.shape[0],
                        image_np_expanded.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')

                # et_time = datetime.now()
                # duration =
                # print("session loaded in ... ", et_time - st_time)
                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={
                                           image_tensor: np.expand_dims(
                                               image_np_expanded,
                                               0)})
                # ed_time = datetime.now()
                # duration = ed_time - st_time
                # print("detection done in ... ", duration)
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

                vis_bool = True if self.json_data['visualization'] == "True" else False

                detection_details_lst = self.get_detection_details(
                    image_np_expanded,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    self.category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    visualize_mask=vis_bool)

                return detection_details_lst

    def publish_rois(self, timestamp, detection_details_lst):
        if detection_details_lst:
            # print("Total ROIs :: ", timestamp, detection_details_lst)
            track_id = 100
            lbl_response_obj = AlgoInterface_pb2.LabelResponse()
            lbl_response_obj.timestamp = timestamp
            #     [{'class': class_name, 'score': score, 'xmin': left,
            #      'xmax': right, 'ymin': top, 'ymax': bottom,
            #      'mask': mask_layer_string})]
            for evry_detect_dict in detection_details_lst:
                prnt_dict = {i: evry_detect_dict[i] for i in evry_detect_dict if i != 'mask'}
                print("label info :> ", prnt_dict)
                nextattr_obj = lbl_response_obj.NextAttr.add()
                nextattr_obj.type.object_class = evry_detect_dict['class']
                nextattr_obj.trackID = track_id
                nextattr_obj.mask = evry_detect_dict['mask']
                print("track id :: ", track_id)
                track_id += 1
                # Create ROI object for Xmin, Ymin
                roi_min_obj1 = nextattr_obj.ROI.add()
                roi_min_obj1.X = int(evry_detect_dict['xmin'])
                roi_min_obj1.Y = int(evry_detect_dict['ymin'])

                roi_min_obj2 = nextattr_obj.ROI.add()
                roi_min_obj2.X = int(evry_detect_dict['xmax'])
                roi_min_obj2.Y = int(evry_detect_dict['ymin'])

                # Create ROI object for Xmax, Ymax
                roi_max_obj3 = nextattr_obj.ROI.add()
                roi_max_obj3.X = int(evry_detect_dict['xmax'])
                roi_max_obj3.Y = int(evry_detect_dict['ymax'])

                roi_max_obj4 = nextattr_obj.ROI.add()
                roi_max_obj4.X = int(evry_detect_dict['xmin'])
                roi_max_obj4.Y = int(evry_detect_dict['ymax'])

            self.lt5_img_publr_obj.send(lbl_response_obj.SerializeToString())

        else:
            print("No ROIs detected..")

            lbl_response_obj = AlgoInterface_pb2.LabelResponse()
            nextattr_obj = lbl_response_obj.NextAttr.add()
            nextattr_obj.type.object_class = 'car'
            nextattr_obj.trackID = 100
            # Create ROI object for Xmin, Ymin
            roi_min_obj1 = nextattr_obj.ROI.add()
            roi_min_obj1.X = -1
            roi_min_obj1.Y = -1

            roi_min_obj2 = nextattr_obj.ROI.add()
            roi_min_obj2.X = -1
            roi_min_obj2.Y = -1

            # Create ROI object for Xmax, Ymax
            roi_max_obj3 = nextattr_obj.ROI.add()
            roi_max_obj3.X = -1
            roi_max_obj3.Y = -1

            roi_max_obj4 = nextattr_obj.ROI.add()
            roi_max_obj4.X = -1
            roi_max_obj4.Y = -1
            self.lt5_img_publr_obj.send(lbl_response_obj.SerializeToString())

    def publ_detection_result(self, topic_name, msg, time):
        global ALGO_READINESS
        ALGO_READINESS = False
        ld_req_obj = imageservice_pb2.ImageResponse()
        lbl_response_obj = AlgoInterface_pb2.LabelResponse()
        if msg is not None:
            ld_req_obj.ParseFromString(msg)
            # print("ld_req_obj :: ", ld_req_obj)
            img_data_obj = ld_req_obj.base_image
            timestamp_img = ld_req_obj.recieved_timestamp
            print("recieved_timestamp :: ", timestamp_img)
            img_data_str = img_data_obj
            nparr = np.fromstring(img_data_str, np.uint8)
            # print("nparr :: ", nparr)
            re_img_np_ary = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # print("re_img_np_ary :: ", re_img_np_ary)
            img_shape = re_img_np_ary.shape
            # print("img_shape ::{ ", img_shape)
            detection_details_lst = self.detect_obj(re_img_np_ary, img_shape[1], img_shape[0])

            self.publish_rois(timestamp_img, detection_details_lst)

    def abort_algo(self, topic_name, msg, time):

        if topic_name == self.json_data['algo_end_response']:
            global BEXIT
            BEXIT = False
                # ecal.finalize()
            # exit(0)
            # print(">>>>>>>>>>>>>>>>", BEXIT)


    def inform_model_loaded(self):
        # Inform model is loaded
        # time.sleep(2)

        lbl_response_obj = AlgoInterface_pb2.AlgoState()
        lbl_response_obj.isReady = True
        self.lt5_algo_publr_obj.send(lbl_response_obj.SerializeToString())

    def define_subscr_callbacks(self):

        # For Image data
        self.lt5_img_subscr_obj.set_callback(self.publ_detection_result)
        self.lt5_finl_subscr_obj.set_callback(self.abort_algo)
        while ecal.ok() and BEXIT:
            # print("#########################", BEXIT)
            time.sleep(0.1)
            if ALGO_READINESS:
                self.inform_model_loaded()
#
# exit(0)
if __name__ == "__main__":
    DetectionMasksROI()
