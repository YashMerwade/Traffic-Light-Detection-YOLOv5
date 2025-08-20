# Traffic-Light-Detection-YOLOv5/v8/v10/v12
Traffic Light detection using YOLO object detector

### Detect objects in both images and video streams using Deep Learning, OpenCV, and Python.

Accurate and real-time detection of traffic lights and their states (Red, Yellow, Green) is essential for intelligent transportation systems and autonomous vehicles. Existing detection models often struggle in complex urban environments due to varying lighting conditions, small object size, occlusion, and inconsistent traffic light designs.

The COCO dataset consists of 3 labels:
- RED
- YELLOW
- GREEN

### Motivation / Why This Project Exists

Traffic lights play a crucial role in regulating traffic flow and ensuring road safety. However, traditional traffic management systems often rely on manual monitoring or simple sensors, which may not be accurate in real-time complex scenarios.

This project was created to address the need for automatic and reliable traffic light detection using deep learning. By leveraging YOLO (You Only Look Once) object detection models such as YOLOv5, YOLOv8, YOLOv10, and YOLOv12, the system can detect traffic lights with high accuracy and classify their states (Red, Yellow, Green).

The main motivation behind this project is to:
- Improve autonomous driving systems with robust traffic light recognition.
- Assist in smart city traffic management by providing real-time monitoring.
- Gain hands-on experience in computer vision and YOLO-based object detection.

Ultimately, the project aims to build a scalable, efficient, and accurate solution for real-world traffic signal detection.

## YOLO object detection in images

## Installation

- `pip install numpy`
- `pip install opencv-python`

## To Run the project

- `python yolo.py --image images/baggage_claim.jpg`

## Screenshots
![Image](/Object%20dection%20using%20image/1.png)

Here you can see that YOLO has detected traffic light in the input image.


## YOLO object detection in video streams

## Installation

- `pip install numpy`
- `pip install opencv-python`

## To Run the project

- `python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco`

## Video Output

![Traffic Light Detection](https://github.com/YashMerwade/Traffic-Light-Detection/blob/main/Traffic Light Detection.mp4?raw=true)


In the video/GIF, you can see traffic light detected.

The YOLO object detector is performing quite well here. 

