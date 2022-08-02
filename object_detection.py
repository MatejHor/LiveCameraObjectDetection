#!/bin/usr/python3
import torch
import numpy as np
import cv2
from time import time
from datetime import datetime

import argparse
from enum import Enum

class Model(Enum):
    n = 'yolov5n'
    s = 'yolov5s'
    m = 'yolov5m'
    l = 'yolov5l'
    x = 'yolov5x'

    def __str__(self):
        return self.value

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, exit='q', model='yolov5s', create_file=False, result_frame=30):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param output: A output file name with path.
        """
        self.model = self.load_model(model)
        self.classes = self.model.names

        self.exit = exit
        self.output = 'output' + str(datetime.timestamp(datetime.now())) + '.avi' if create_file else None
        self.frame = result_frame
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, model):
        """
        Loads Yolo5 model from pytorch hub.
        :param model: name of model to use 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', str(model), pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        # Load camera stream
        camera = cv2.VideoCapture(0)
        assert camera.isOpened()

        # Get image shape
        image_shape = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Create video writer
        if self.output:
            four_cc = cv2.VideoWriter_fourcc(*"MJPG")
            output = cv2.VideoWriter(self.output, four_cc, self.result_frame, image_shape) 

        while True:
            start_time = time()
            check, frame = camera.read()
            assert check

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()

            fps = 1/np.round(end_time - start_time, 3)
            print(f"[+] FPS: {fps}", end='\r')

            # Write result into file
            if self.output: output.write(frame)

            # End video window
            cv2.imshow('CameraVideo', frame)
            if self.exit == 'x':
                if cv2.waitKey(1):
                    if cv2.getWindowProperty('CameraVideo', cv2.WND_PROP_VISIBLE) < 1:
                        break
            if cv2.waitKey(1) & 0xFF == ord(self.exit):
                break

        camera.release()
        cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', help='Store result into file', default=False, action='store_true')
parser.add_argument('-f', '--fps', help='FPS for output file', default=30)
parser.add_argument('--model', help='Model which script can use', default='yolov5s', type=Model, choices=list(Model))
parser.add_argument('--exit', help='Key to exit program', default='q')


args = parser.parse_args()
object_detection = ObjectDetection(
    model=args.model, 
    create_file=args.output, 
    result_frame=args.fps, 
    exit=args.exit
    )
object_detection()
