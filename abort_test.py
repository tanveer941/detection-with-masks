

import ecal
import imageservice_pb2
import AlgoInterface_pb2
import sys
import time

ecal.initialize(sys.argv, "HFL data requestor")
# ecal.initialize(sys.argv, "H5 data publisher")
# hfl_publ_obj = ecal.publisher("Request_Device")
# hfl_publ_obj = ecal.publisher("Request_Channel")
hfl_publ_obj = ecal.publisher("Finalize")

time.sleep(2)
hfl_req_proto_obj = imageservice_pb2.ImageRequest()
# hfl_dvc_proto_obj = imageservice_pb2.deviceTypeRequest()
hfl_chnl_proto_obj = AlgoInterface_pb2.AlgoState()

# def request_device_typ():
#     hfl_dvc_proto_obj.device_type = "dfhxzfxgjhgcj"
#     # time.sleep(2)
#
#     hfl_publ_obj.send(hfl_dvc_proto_obj.SerializeToString())
#     print "hfl_req_proto_obj.SerializeToString() :: ", hfl_dvc_proto_obj.SerializeToString()
#
#     ecal.finalize()

def req_channel_type():
    hfl_chnl_proto_obj.required_devicesData = True
    payload = hfl_chnl_proto_obj.SerializeToString()
    # print "payload :: ", payload
    hfl_publ_obj.send(payload)

    ecal.finalize()

def req_image_data():
    hfl_chnl_proto_obj.isReady = True
    payload = hfl_chnl_proto_obj.SerializeToString()
    # print "payload :: ", payload
    hfl_publ_obj.send(payload)

    ecal.finalize()

# request_device_typ()
# req_channel_type()
req_image_data()
