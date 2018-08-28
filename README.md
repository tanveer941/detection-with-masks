# detection-with-masks
Tensorflow object detection with image segmentation
# Object detection algorithm trained on COCO dataset with SSD network.
  - An eCAL layer has been integrated over it to seamlessly allow the passage of input and output.
  - The exchange of messages happen over eCAL through protobuf
  - input protobuf message is defined in imageservice.proto
  - output protobuf message is defined in AlgoInterface.proto
  - The pre-trained model weights is in the repo as a .pb file
  - The latest model can be downloaded from repo and replaced as it is in the directory.
  - object_class_filter.json will have the list of objects that needs to be detected and published over eCAL

One can refer to object_classes.txt file for list of object classes the model has been pre-trained for.
# Breakdown of topics.json:
  - image_request: topic name on which the image data is subscribed
  - image_response: topic name on which the output data is published(Bounding box coordinates, class names and image numpy array)  
  - algo_begin_response: topic name on which a notification is sent across to other process to notify that the model has been loaded successfully.
  - algo_end_response: topic name on which the entire application is terminated and eCAL stops polling for a message.
  - visualization: If set to 'True' then an image would pop-up with segmentations over the image and the respective objects
               being identified by their class names

  - Compiling proto files : protoc -I=.\ --python_out=.\ AlgoInterface.proto  
        eCAL 4.9.0, 4.9.3 and 4.9.6 has been tested with label tool 5G

# Inference through eCAL - How to use?
  - In topics.json, set the topic names for "image_request" and "image_response".  
  - In topics.json, set the topic names for "algo_begin_response" and "algo_end_response". 
  - On the other end, the image must be published on the topic name as defined in the image_request.  
  - The box coordinates, class name and the mask array should be published on the topic name defined in image_response.  
  - Run the file rcnn_detection_service.py. Wait until the model is loaded. Once done, publish images on the other end. 
  - If visualization is set to "True" in topics.json it would pop an image with masks and classes marked on it. 
  - To enable the detection algo to continuosly publish the output over eCAL through protobuf, set visualization to "False".  

# Executable creation
  - Once .exe is created, copy the model weights file into the directory. 
  - Copy topics.json, object_classes.txt and object_class_filter.json 
  - C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile mask_ecal_detect.py --add-data _ecal_py_3_5_x64.pyd;. 
  - Using the '--onefile' attribute greatly reduces the size of .exe by half of it eliminating the inclusion of unnecessary DLLs. 
