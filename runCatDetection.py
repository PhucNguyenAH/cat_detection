import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import os, requests, math
import numpy as np
import PIL
if os.getcwd().split("/")[-1]=="wowAI":
  os.chdir("src")
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional


def check_img_size(img_size, s=32, floor=0):
    def make_divisible( x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

def precess_image(img_src, img_size, stride, half):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image

if __name__ == "__main__":
    checkpoint:str = "runs/train/yolov6s6_cat/weights/best_ckpt"
    half:bool = False #@param {type:"boolean"}

    if not os.path.exists(f"{checkpoint}.pt"):
        print("No checkpoint...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.set_page_config(layout="wide") 
    st.title("CAT DETECTION")  
    
    if "model" not in st.session_state:
        model = DetectBackend(f"./{checkpoint}.pt", device=device)
        st.session_state.model = model
    else:
        model = st.session_state.model
    stride = model.stride
    class_names = load_yaml("./data/dataset.yaml")['names']

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False
    hide_labels: bool = False #@param {type:"boolean"}
    hide_conf: bool = False #@param {type:"boolean"}

    img_size:int = 640#@param {type:"integer"}

    conf_thres: float =.5 #@param {type:"number"}
    iou_thres: float =.45 #@param {type:"number"}
    max_det:int =  1000#@param {type:"integer"}
    agnostic_nms: bool = False #@param {type:"boolean"}
    img_size = check_img_size(img_size, s=stride)
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup
   
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        if st.button('Detect cat'):
            img_src = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

            img_size = check_img_size(img_size, s=stride)

            img = precess_image(img_src, img_size, stride, half)
            img = img.to(device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            pred_results = model(img)
            classes:Optional[List[int]] = None # the classes to keep
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src.copy()
            if len(det):
                det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
            # PIL.Image.fromarray(img_ori)
            st.image(img_ori)

    else:
        path = "img_test/none.jpg"
        img_src = cv2.imread(path)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        st.image(img_src)