

# Traffic-Light-Detection-YOLOv5/v8/v10/v12

Accurate and real-time detection of traffic lights and their states (Red, Yellow, Green) is essential for intelligent transportation systems and autonomous vehicles. Existing detection models often struggle in complex urban environments due to varying lighting conditions, small object size, occlusion, and inconsistent traffic light designs.

We trained our yolo models on google colab so there are some steps
to check the GPU and install the requirments to train the specific model.


## Check GPU availability

NOTE: YOLOv12 leverages FlashAttention to speed up attention-based computations, but this feature requires an Nvidia GPU built on the Ampere architecture or newer—for example, GPUs like the RTX 3090, RTX 3080, or even the Nvidia L4 meet this requirement.
To set GPU in runtime change runtime to your availabel GPU.
and run this below code to check it.

```bash
  !nvidia-smi
```
```bash
  import os
HOME = os.getcwd()
print(HOME)
```
## Install dependencies
NOTE: Currently, YOLOv12 does not have its own PyPI package, so we install it directly from GitHub while also adding roboflow (to conveniently pull datasets from the Roboflow Universe), supervision (to visualize inference results and benchmark the model’s performance), and flash-attn (to accelerate attention-based computations via optimized CUDA kernels).
```bash
  !pip install -q git+https://github.com/sunsmarterjie/yolov12.git roboflow supervision flash-attn
```
If you are training different model like yolov10 install there dependencies.
## Download dataset from Roboflow Universe
This code u can get when downlod a dataset roboflow where you can select the option to generate code that code you have to paste in colab and run the code the dataset directly downlod in your colab.
```bash
  !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xB4CL8mIant14XpYAKLH")
project = rf.workspace("traffic-light-4a7sq").project("traffic-light-pu1o5-lknqx")
version = project.version(1)
dataset = version.download("yolov12")
```
```bash
!ls {dataset.location}
```
NOTE: We need to make a few changes to our downloaded dataset so it will work with YOLOv12. Run the following bash commands to prepare your dataset for training by updating the relative paths in the data.yaml file, ensuring it correctly points to the subdirectories for your dataset's train, test, and valid subsets.
```bash
!sed -i '$d' {dataset.location}/data.yaml
!sed -i '$d' {dataset.location}/data.yaml
!sed -i '$d' {dataset.location}/data.yaml
!sed -i '$d' {dataset.location}/data.yaml
!echo -e "test: ../test/images\ntrain: ../train/images\nval: ../valid/images" >> {dataset.location}/data.yaml
```
```bash
!cat {dataset.location}/data.yaml
```
## Fine-tune YOLO model(train the model)
We are now ready to fine-tune our YOLOv12 model. In the code below, we initialize the model using a starting checkpoint—here, we use yolov12s.yaml, but you can replace it with any other model (e.g., yolov12n.pt, yolov12m.pt, yolov12l.pt, or yolov12x.pt) based on your preference. We set the training to run for 100 epochs in this example; however, you should adjust the number of epochs along with other hyperparameters such as batch size, image size, and augmentation settings (scale, mosaic, mixup, and copy-paste) based on your hardware capabilities and dataset size
```bash
from ultralytics import YOLO

model = YOLO('yolov12s.yaml')

results = model.train(data=f'{dataset.location}/data.yaml', epochs=100)
```
## Evaluate fine-tuned YOLO model

Here we can generate the confusion matrix,train and val losses by running these below codes.
```bash
import locale
locale.getpreferredencoding = lambda: "UTF-8"

!ls {HOME}/runs/detect/train/
```
Confusion matrix
```bash
from IPython.display import Image

Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=1000)
```
Results of trained model
```bash
from IPython.display import Image

Image(filename=f'{HOME}/runs/detect/train/results.png', width=1000)
```
To check mean average precision of model
```bash
import supervision as sv

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{dataset.location}/test/images",
    annotations_directory_path=f"{dataset.location}/test/labels",
    data_yaml_path=f"{dataset.location}/data.yaml"
)

ds.classes
```
```bash
from supervision.metrics import MeanAveragePrecision

model = YOLO(f'/{HOME}/runs/detect/train/weights/best.pt')

predictions = []
targets = []

for _, image, target in ds:
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    predictions.append(detections)
    targets.append(target)

map = MeanAveragePrecision().update(predictions, targets).compute()
```
```bash
print("mAP 50:95", map.map50_95)
print("mAP 50", map.map50)
print("mAP 75", map.map75)
```
We can plot the results by map.plot()

## Run inference with fine-tuned YOLO model
Here we can test our trained model by using test set images.
```bash
import supervision as sv

model = YOLO(f'/{HOME}/runs/detect/train/weights/best.pt')

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{dataset.location}/test/images",
    annotations_directory_path=f"{dataset.location}/test/labels",
    data_yaml_path=f"{dataset.location}/data.yaml"
)
```
```bash
import random

i = random.randint(0, len(ds))

image_path, image, target = ds[i]

results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results).with_nms()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = image.copy()
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
```
run above code it will test random test images give you output.

## Deployment trained model
If you want to test your model on video or images you run this below code,in model="yourmodel.pt" and source="img.jpg" or "video.mp4".
```bash
!yolo task=detect mode=predict model="/content/best.pt" conf=0.25 source="/content/traffic2.jpg" save=True
```
These are steps if you follow them you can train model it could vary little bit for differnt yolo models. 