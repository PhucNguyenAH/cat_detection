import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import os, requests, math
import numpy as np
import PIL
import onnxruntime as ort
import random

if os.getcwd().split("/")[-1]=="wowAI":
  os.chdir("src")
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

if __name__ == "__main__":
    checkpoint:str = "runs/train/yolov6s6_cat/weights/best_ckpt.onnx"
    cuda = True
    resize_data = []

    if not os.path.exists(checkpoint):
        print("No checkpoint...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.set_page_config(layout="wide") 
    st.title("CAT DETECTION")  
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    if "session" not in st.session_state:
        session = ort.InferenceSession(checkpoint, providers=providers)
        st.session_state.session = session
    else:
        session = st.session_state.session
    

    names = load_yaml("./data/dataset.yaml")['names']
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
   
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        if st.button('Detect cat'):
            origin_RGB = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            image = origin_RGB.copy()
            image, ratio, dwdh = letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            im = image.astype(np.float32)
            resize_data.append((im,ratio,dwdh))
            np_batch = np.concatenate([data[0] for data in resize_data])
            outname = [i.name for i in session.get_outputs()]
            inname = [i.name for i in session.get_inputs()]
            im = np.ascontiguousarray(np_batch/255)
            out = session.run(outname,{'images':im})
            image = origin_RGB
            num = out[0][0][0]
            ratio,dwdh = resize_data[0][1:]
            for obj in range(num):
                x0,y0,x1,y1 = out[1][0][obj]
                score = out[2][0][obj]
                cls_id = out[3][0][obj]
                box = np.array([x0,y0,x1,y1])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                cls_id = int(cls_id)
                score = round(float(score),3)
                if score <= 0.5:
                    continue
                name = names[cls_id]
                color = colors[name]
                name += ' '+str(score)
                cv2.rectangle(image,box[:2],box[2:],color,2)
                cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0, 0, 0],thickness=2)
            st.image(origin_RGB)

    else:
        path = "img_test/none.jpg"
        img_src = cv2.imread(path)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        st.image(img_src)