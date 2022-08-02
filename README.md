# Object detection with YOLOv5 and OpenCV

Simple python script which detects objects using the YOLOv5 algorithm from the camera.
Camera video is provided by OpenCV and object detection model (YOLOv5) by PyTorch.
The script will automatically download the YOLOv5 model, and also is possible to choose which model will be used. 
Available models:
- 'yolov5n'
- 'yolov5s'
- 'yolov5m'
- 'yolov5l'
- 'yolov5x'

## Install
```shell
python -m venv ./ob_yolo
source ./ob_yolo/bin/activate # or for windows .\ob_yolo\Scripts\activate.bat
pip install -r requirements.txt
```

## Usage
```shell
python object_detection.py 
```

Also is it possible to store the whole video from object detection with arg parameter `-o, --output <file_name>` and define how much fps video will have `-f, --fps <number>, default 30`.
To use a different model than `yolov5s` use arg parameter `--model <model_name>` and choose a model name from the list above.
And it is possible to define which key should end the camera program, the default is the `q` key and to redefine use arg parameter `--exit <key>`.

