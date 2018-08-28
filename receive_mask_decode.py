
import ecal
import sys
import AlgoInterface_pb2
import numpy as np
import cv2

def subscribe_signl_names():

    ecal.initialize(sys.argv, "Detector mask subscriber")
    subscribe_sig_names_obj = ecal.subscriber(topic_name="Pixel_Response")
    rdr_sig_response = AlgoInterface_pb2.LabelResponse()

    while ecal.ok():
        ret, msg, time = subscribe_sig_names_obj.receive(500)
        print("---:: ", ret, msg, time, type(msg))
        if msg is not None:
            rdr_sig_response.ParseFromString(msg)
            object_attibute_lst = rdr_sig_response.NextAttr
            # print("object_attibute :: ", object_attibute_lst)

            for evry_obj_attr in object_attibute_lst:
                track_id = evry_obj_attr.trackID
                # print("track_id ::> ", track_id)
                class_obj = evry_obj_attr.type.object_class
                # print("class_obj :: ", class_obj)
                x1 = evry_obj_attr.ROI[0].X
                y1 = evry_obj_attr.ROI[0].Y

                x2 = evry_obj_attr.ROI[1].X
                y2 = evry_obj_attr.ROI[1].Y

                x3 = evry_obj_attr.ROI[2].X
                y3 = evry_obj_attr.ROI[2].Y

                x4 = evry_obj_attr.ROI[3].X
                y4 = evry_obj_attr.ROI[3].Y

                print("ordinates :: ", x1, x2, x3, x4, y1, y2, y3, y4)

                mask = evry_obj_attr.mask
                with open('img.txt', 'w+') as fhandle:
                    fhandle.write(str(mask))

                nparr = np.fromstring(mask, np.uint8)

                print("nparr :: ", nparr)
                re_img_np_ary = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_shape = re_img_np_ary.shape
                print("shape::", img_shape)
                cv2.imwrite('color_img.jpg', re_img_np_ary)
                cv2.imshow('Color image', re_img_np_ary)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



subscribe_signl_names()