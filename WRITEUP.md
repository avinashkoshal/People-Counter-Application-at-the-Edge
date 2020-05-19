# Project Write-Up

This document provides my project write-up .i.e information on different things and 
approaches that I have used

## Introduction about project
   The people counter application demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. The application detects people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count. 

   For this porect I have used Intel OpenVINO Toolkit.
   Two main components of Intel OpenVINO Toolkit are : " Model Optimizer and Inference Engine "

## How the application works?

	The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used is be able to identify people in a video frame. The application counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.

## Explaining Custom Layers in OpenVINO™

   Custom layers are layers that are not included in the list of known layers. If topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.
   They cannot be directly handled in OpenVino. 
   When implementing a custom layer for pre-trained model in the Intel® Distribution of OpenVINO™ toolkit, you will need to add extensions to both the Model Optimizer and the Inference Engine.

   To create a custom layer for OpenVino, you must add extensions to the Model Optimizer and the Inference Engine. For this, the first step is to use the Model Extension Generator tool The MEG is going to create templates for Model Optimizer extractor extension, Model Optimizer operations extension, Inference Engine CPU extension and Inference Engine GPU extension. Once customized the templates, next step is to generate the IR files with the Model Optimizer. Finally, before using the custom layer in your model with the Inference Engine, you must: first, edit the CPU extension template files, second, compile the CPU extension and finally, execute the model with the custom layer.

### Custom Layer Extensions for the Model Optimizer
    The Model Optimizer first extracts information from the input model which includes the topology of the model layers along with parameters, input and output format, etc., for each layer. The model is then optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Finally, the optimized model is output to the model IR files needed by the Inference Engine to run the model.

	The Model Optimizer starts with a library of known extractors and operations for each supported model framework which must be extended to use each unknown custom layer. The custom layer extensions needed by the Model Optimizer are:
	     - Custom Layer Extractor
	     - Custom Layer Operation

### Custom Layer Extensions for the Inference Engine
    Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

	   - Custom Layer CPU Extension
	         A compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.
	   - Custom Layer GPU Extension
	         OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.

## Model Selection

	TensorFlow Object Detection Model Zoo contains many pre-trained models on the coco dataset. For this project, various classes of models were tested from the TensorFlow Object Detection Model Zoo. SSD_inception_v2_coco and faster_rcnn_inception_v2_coco performed good as compared to rest of the models, but, in this project, faster_rcnn_inception_v2_coco is used which is fast in detecting people with less errors. Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.

## Comparing Model Performance
   There is huge difference on running Plain model and on OpenVino

### Model-1: Ssd_inception_v2_coco_2018_01_28(OpenVINO)          |  Ssd_inception_v2_coco_2018_01_28(Tensorflow)
                                                                 |
#### Latency(microseconds) = 115                                 |  Latency(microseconds) = 222
#### Memory(mb) = 329                                            |  Memory(mb) = 538
	                                                             |
### Model-2: Faster_rcnn_inception_v2_coco_2018_01_289(OpenVINO) |  Faster_rcnn_inception_v2_coco_2018_01_289(Tensorflow)
                                                                 |
#### Latency(microseconds) = 889                                 |  Latency(microseconds) = 1281
#### Memory(mb) = 281                                            |  Memory(mb) = 562

## Assess Effects on End User Needs
   Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. Lighting, focal length a,d image size are relevant to system behavior, a bad lighting can decrease model performance by diffusing image info, image size is relevant although YOLO models are quite size independent and the same happens with focal length. Camera vision angle is also relevant for this kind of tasks and for performance of the system. Depending on the dataset used, (COCO in this model) some kind of angles can decrease model accuracy and also increase number of occlusions with the problems this generates in detection. Model accuracy is relevant due to the amount of false positives or negatives it can generate degrading system performance.

## Model Use Cases

	This application could keep a check on the number of people in a particular area and could be helpful where there is restriction on the number of people present in a particular area. This model could be uisng in Shopping malls, cinema halls and temples, railway and places where only certain number of persons are allowed.



## Application Requirements

### Hardware Requirements

   - 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
   - OR use of Intel® Neural Compute Stick 2 (NCS2)

### Software Requirements

   - Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
   - Node v6.17.1
   - Npm v3.10.10
   - CMake
   - MQTT Mosca server
   - Python 3.5 or 3.6

## Setup
   - Install Intel® Distribution of OpenVINO™ toolkit
   - Install Nodejs and its dependencies
   - Install npm

   There are three components that need to be running in separate terminals for this application to work:

   - MQTT Mosca server
   - Node.js* Web server
   - FFmpeg server

   To start the three servers in separate terminal windows, following commands should be executed from the main directory:

	    For MQTT/Mosca server:

		   $cd webservice/server
		   $npm install

	    For Web server:

		   $cd ../ui
		   $npm install

	    For FFmpeg Server:

		   $sudo apt install ffmpeg