import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser



def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)

    w = int(cap.get(3))
    h = int(cap.get(4))

    in_shape = network_shape['image_tensor']

    #iniatilize variables
    
    duration_previous = 0
    counter_total = 0
    dur = 0
    request_id=0
    
    report = 0
    counter = 0
    counter_previous = 0
    
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        duration_report = None
        infer_network.exec_net(net_input, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
         
            pointer = 0
            probs = net_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    pointer += 1
                    box = net_output[0, 0, i, 3:]
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
        
            if pointer != counter:
                counter_previous = counter
                counter = pointer
                if dur >= 3:
                    duration_previous = dur
                    dur = 0
                else:
                    dur = duration_previous + dur
                    duration_previous = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > counter_previous:
                        counter_total += counter - counter_previous
                    elif dur == 3 and counter < counter_previous:
                        duration_report = int((duration_previous / 10.0) * 1000)

            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
 

        ### TODO: Send the frame to the FFMPEG server ###
        #  Resize the frame
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
