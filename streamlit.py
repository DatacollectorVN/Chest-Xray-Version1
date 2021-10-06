import streamlit as st

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import warnings

warnings.filterwarnings("ignore")


import cv2
from PIL import Image,ImageEnhance

from Keras_retinanet.util.util_1 import *
import time

from Keras_retinanet.keras_retinanet import models
from Keras_retinanet.keras_retinanet.utils.image import  preprocess_image, resize_image

from Keras_retinanet.keras_retinanet.utils.colors import label_color

def img_inference(model,image, THRES_SCORE, label_name_id):
        # image = read_image_bgr(img_path)

        image = Image.open(image)
        image = np.array(image.convert("RGB"))
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        duration = time.time() - start

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < THRES_SCORE:
                break

            color = label_color(label)
            caption = "{} {:.3f}".format(label_name_id[label_name_id[1] == label][0].values[0], score)

            b = box.astype(int)
            draw = draw_bbox(draw, list(np.int_(b)),
                                caption,
                                color=color,
                                thickness=3)
        return draw, duration


def main():
    """Run this function to display the Streamlit app"""

    st.header("***Chest X-ray abnormalities detection (DEMO)***")
    st.write("Code by Nhan Ngo")
    st.title("Diseases Dectection")
    st.write("If you don't have the X-ray image, click this link below to download:")
    st.write("[link_to_download_sample_image](https://drive.google.com/drive/folders/1JzrJ0cpqFvUI1t6saBgK1xayz1VxgIPO?usp=sharing)")
    confidence_threshold = st.sidebar.selectbox(
        'Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?',
        (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1))
    st.markdown('''***5 diseases:***\n
    + Aortic enlargement
    + Cardiomegaly
    + ILD
    + Pleural thickening
    + Pulmonary fibrosis''')
    file = st.file_uploader('Upload img file (JPG/PNG format)')
    st.header("Image before")
    show_file_1 = st.empty()

    if not file:
        show_file_1.info("Please upload a file of type: jpg, png")
        return

    show_file_1.image(file)

    
    weighted_85_5_classes = "Keras_retinanet/snapshots/pretrain_model.h5"
    model = models.load_model(weighted_85_5_classes,
                              backbone_name='resnet101')
    model = models.convert_model(model)
    label_name_id = pd.read_csv("Keras_retinanet/classes_5.csv", header=None)

    img_new, duration  = img_inference(model, file, confidence_threshold, label_name_id)
    st.write("Duration:", duration)
    st.header("Image After")
    show_file_2 = st.empty()
    show_file_2.image(img_new)
    file.close()

main()
