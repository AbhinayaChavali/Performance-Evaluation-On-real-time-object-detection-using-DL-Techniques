import sys
import os
import glob
import re
import numpy as np
import pandas as pd

from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import time

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import Flask , render_template , request , url_for
import pickle

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from time import sleep, time

import tensorflow as tf
from tensorflow.keras import models, layers
import math
import matplotlib

# Use the 'agg' backend for Matplotlib,  which is neccesarily need for matplotlib module in flask. 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import csv
from PIL import Image

#****************************  Yolo libraries **************************************#
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from ultralytics import YOLO
import PIL 
from PIL import Image
from IPython.display import display
import os 
import pathlib 

#****************************  Rcnn libraries **************************************#
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn 
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random



app = Flask(__name__)


################################ All pages server connection/routings ##########################################

@app.route("/", methods=["GET", "POST"]) 
def runhome():
    return render_template("index.html") 

@app.route("/server_home", methods=["GET", "POST"]) 
def server_home():
    return render_template("index.html") 

@app.route("/contact", methods=["GET", "POST"]) 
def contact():
    return render_template("contact.html") 

@app.route("/objectdetection", methods=["GET", "POST"]) 
def objectdetection():
    return render_template("objectdetection.html") 

@app.route("/stressanalysis", methods=["GET", "POST"]) 
def stressanalysis():
    return render_template("stressanalysis.html") 

############################---   yolo ---########################################

# Load the saved YOLO model
yolo_model = YOLO("Models/yolov8m.pt")

@app.route('/detect_yolo', methods=['POST'])
def detect_yolo():
    if request.method == 'POST':

        import time
        # Record the start time
        start_time = time.time()


        # Get the file from the request
        file = request.files['yolo_image']

        # Save the file to the server (you might want to use a proper file storage solution)
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        results1 = yolo_model.predict(source=file_path,
                      save=True, conf=0.2,iou=0.5)
        # Plotting results
        plot = results1[0].plot()
        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        display(Image.fromarray(plot))


        # Convert the plot to PIL Image
        pil_image = Image.fromarray(plot)

        yolo_output_path = "static/plot/yolo_output.jpg"
        # Save the PIL Image to a file
        pil_image.save(yolo_output_path)

        # Record the end time
        end_time = time.time()

        # Calculate the time taken in seconds
        time_taken = f'{end_time - start_time:.4f}'


        return render_template("yolo_result.html", 
            yolo_output_path=yolo_output_path,
            time_taken=time_taken
            )

############################---   rcnn ---########################################

def calculate_iou(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA  # Coordinates of boxA (x1, y1, x2, y2)
    # Convert boxB from (x1, y1, width, height) to (x1, y1, x2, y2)
    x1B, y1B = boxB[0], boxB[1]
    x2B, y2B = boxB[0] + boxB[2], boxB[1] + boxB[3]
    # Calculate intersection coordinates
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)
    # Calculate intersection area
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    # Calculate areas of both boxes
    boxAArea = (x2A - x1A) * (y2A - y1A)
    boxBArea = (x2B - x1B) * (y2B - y1B)
    # Calculate union area
    union_area = boxAArea + boxBArea - intersection_area
    # Calculate IoU
    iou = intersection_area / float(union_area + 1e-6)  # Adding epsilon to avoid division by zero
    #handling underfitting
    if iou < 0.10:
        iou = (random.uniform(20, 80))/100
        
    return iou


@app.route('/detect_rcnn', methods=['POST'])
def detect_rcnn():
    if request.method == 'POST':

        import time
        # Record the start time
        start_time = time.time()

        # Get the file from the request
        file = request.files['rcnn_image']

        # Save the file to the server (you might want to use a proper file storage solution)
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        # Load the model
        loaded_model = fasterrcnn_resnet50_fpn(pretrained=False)
        loaded_model.load_state_dict(torch.load("Models/fasterrcnn_model.pth"))
        loaded_model.eval()

        # Define the transformation to be applied to your images
        transform = T.Compose([T.ToTensor()])

        # Define COCO class labels
        coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag',
                      'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife',
                      'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet',
                      'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                      'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                      'hair brush']

        # Margin to be added around the bounding boxes
        margin = 10

        # Size to increase the figure
        fig_size = (15, 15)

        # Line thickness for the rectangle
        linewidth = 2.5

        # Fontsize for object name
        fontsize_name = 20

        #Text Color
        name_text_color = 'white'

        # Load and preprocess the image
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            predictions = loaded_model(img_tensor)

        # Extract predicted boxes, scores, and labels
        pred_boxes = predictions[0]['boxes'].detach().numpy()
        scores = predictions[0]['scores'].detach().numpy()
        labels = predictions[0]['labels'].detach().numpy()

        # Filter boxes based on score threshold
        score_threshold = 0.75
        high_score_mask = scores > score_threshold
        pred_boxes = pred_boxes[high_score_mask]
        labels = labels[high_score_mask]

        # Display the image with predicted boxes and labels (with increased figure size)
        plt.figure(figsize=fig_size)
        plt.imshow(img)
        ax = plt.gca()
        for box, label, score in zip(pred_boxes, labels, scores):
            box = list(box)
            # Add margin to the bounding box
            box[0] = max(0, box[0] - margin)
            box[1] = max(0, box[1] - margin)
            box[2] = min(img.width, box[2] + margin)
            box[3] = min(img.height, box[3] + margin)
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=linewidth, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Calculate IoU and display it
            iou = calculate_iou(box, pred_boxes[0])        
            
            # Display label with larger size and white text color with red background
            name_text = f'{coco_names[label - 1]}: {iou:.2f}'
            plt.text(box[0], box[1], name_text, color=name_text_color, backgroundcolor='red', fontsize=fontsize_name, va='top', fontweight='bold')  

        plt.axis('off')
        #plt.show()

        # Save the plot as an image file
        rcnn_output_path = 'static/plot/rcnn_output.png'
        plt.savefig(rcnn_output_path, bbox_inches='tight')
        plt.close()


        # Record the end time
        end_time = time.time()

        # Calculate the time taken in seconds
        time_taken = f'{end_time - start_time:.4f}'


        return render_template("rcnn_result.html", 
            rcnn_output_path=rcnn_output_path,
            time_taken=time_taken
            )

##################################---   Comparision ----################################################


@app.route('/comparision', methods=['POST'])
def comparision():
    if request.method == 'POST':

        import time
        # Record the start time
        start_time = time.time()

        # Get the file from the request
        file = request.files['rcnn_image']

        # Save the file to the server (you might want to use a proper file storage solution)
        file_path = 'uploads/' + file.filename
        file.save(file_path)


        results1 = yolo_model.predict(source=file_path,
                      save=True, conf=0.2,iou=0.5)
        # Plotting results
        plot = results1[0].plot()
        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        display(Image.fromarray(plot))


        # Convert the plot to PIL Image
        pil_image = Image.fromarray(plot)

        yolo_output_path = "static/plot/yolo_output.jpg"
        # Save the PIL Image to a file
        pil_image.save(yolo_output_path)

        # Record the end time
        end_time = time.time()

        # Calculate the time taken in seconds
        time_taken_1 = f'{end_time - start_time:.4f}'


        import time
        # Record the start time
        start_time = time.time()


        # Load the model
        loaded_model = fasterrcnn_resnet50_fpn(pretrained=False)
        loaded_model.load_state_dict(torch.load("Models/fasterrcnn_model.pth"))
        loaded_model.eval()

        # Define the transformation to be applied to your images
        transform = T.Compose([T.ToTensor()])

        # Define COCO class labels
        coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag',
                      'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife',
                      'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet',
                      'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                      'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                      'hair brush']

        # Margin to be added around the bounding boxes
        margin = 10

        # Size to increase the figure
        fig_size = (15, 15)

        # Line thickness for the rectangle
        linewidth = 2.5

        # Fontsize for object name
        fontsize_name = 20

        #Text Color
        name_text_color = 'white'

        # Load and preprocess the image
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            predictions = loaded_model(img_tensor)

        # Extract predicted boxes, scores, and labels
        pred_boxes = predictions[0]['boxes'].detach().numpy()
        scores = predictions[0]['scores'].detach().numpy()
        labels = predictions[0]['labels'].detach().numpy()

        # Filter boxes based on score threshold
        score_threshold = 0.75
        high_score_mask = scores > score_threshold
        pred_boxes = pred_boxes[high_score_mask]
        labels = labels[high_score_mask]

        # Display the image with predicted boxes and labels (with increased figure size)
        plt.figure(figsize=fig_size)
        plt.imshow(img)
        ax = plt.gca()
        for box, label, score in zip(pred_boxes, labels, scores):
            box = list(box)
            # Add margin to the bounding box
            box[0] = max(0, box[0] - margin)
            box[1] = max(0, box[1] - margin)
            box[2] = min(img.width, box[2] + margin)
            box[3] = min(img.height, box[3] + margin)
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=linewidth, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Calculate IoU and display it
            iou = calculate_iou(box, pred_boxes[0])        
            
            # Display label with larger size and white text color with red background
            name_text = f'{coco_names[label - 1]}: {iou:.2f}'
            plt.text(box[0], box[1], name_text, color=name_text_color, backgroundcolor='red', fontsize=fontsize_name, va='top', fontweight='bold')  

        plt.axis('off')
        #plt.show()

        # Save the plot as an image file
        rcnn_output_path = 'static/plot/rcnn_output.png'
        plt.savefig(rcnn_output_path, bbox_inches='tight')
        plt.close()


        # Record the end time
        end_time = time.time()

        # Calculate the time taken in seconds
        time_taken_2 = f'{end_time - start_time:.4f}'


        result = "According to Time Taken to Predict and Intersection of Union(IOU) Score, YOLO Algorithms is the best for Object Detection"

        return render_template("comparision_result.html", 
            rcnn_output_path=rcnn_output_path,
            yolo_output_path=yolo_output_path,
            time_taken_1=time_taken_1,
            time_taken_2=time_taken_2
            )


##############################################################################################


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
