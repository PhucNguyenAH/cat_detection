import os
import cv2
from tqdm import tqdm
from glob import glob
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import shutil
from sklearn.model_selection import train_test_split

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    labels = load_labels(path)[1:]

    w,h = img.shape[:2]
    
    return img, labels , w , h
    
def load_labels(path):
    path = path + ".cat"
    
    with open(path,'r') as f:
        coordinates = f.readline()
        coordinates = str(coordinates).split(' ')[:-1]
    
    return list(map(int,coordinates))

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
if not os.path.exists("dogs_data/images/train"):
    os.makedirs("dogs_data/images/train") 
if not os.path.exists("dogs_data/images/val"):
    os.makedirs("dogs_data/images/val") 
if not os.path.exists("dogs_data/labels/train"):
    os.makedirs("dogs_data/labels/train") 
if not os.path.exists("dogs_data/labels/val"):
    os.makedirs("dogs_data/labels/val") 
for dir in glob("dogs/images/Images/*"):
    for file in glob(f"{dir}/*.jpg"):
        shutil.copy(file, os.path.join("dogs_data","images"))

for dir in tqdm(glob("dogs/annotations/Annotation/*")):
    for file in glob(f"{dir}/*"):
        
        filename = file.split("/")[-1]
        with open(file, 'r') as f:
            data = f.read()

        Bs_data = BeautifulSoup(data, "xml")
        xmin_lst = Bs_data.find_all('xmin')
        xmax_lst = Bs_data.find_all('xmax')
        ymin_lst = Bs_data.find_all('ymin')
        ymax_lst = Bs_data.find_all('ymax')
        w = int(Bs_data.find_all('width')[0].text)
        h = int(Bs_data.find_all('height')[0].text)
        f = open(os.path.join("dogs_data",f"{filename}.txt"), "w")
        for i in range(len(xmin_lst)):
            xmin = int(xmin_lst[i].text)
            xmax = int(xmax_lst[i].text)
            ymin = int(ymin_lst[i].text)
            ymax = int(ymax_lst[i].text)
            b = (xmin, xmax, ymin, ymax)
            bb = convert((w,h), b)
            f.write(f"0 {round(bb[0],6)} {round(bb[1],6)} {round(bb[2],6)} {round(bb[3],6)}\n")
        f.close()

if not os.path.exists("dogs_data/train.txt") and not os.path.exists("dogs_data/val.txt"):
    data = []
    for file in glob("dogs_data/*.txt"):
        filename = file.split("/")[-1].split(".")[0]
        data.append(filename)
    train, val = train_test_split(data, test_size=0.2, random_state=42)
    f = open("dogs_data/train.txt", "w")
    for d in train:
        f.write(f"./images/train/{d}.jpg\n")
        shutil.move(os.path.join("dogs_data","images", f"{d}.jpg"), os.path.join("dogs_data","images","train"))
        shutil.move(os.path.join("dogs_data", f"{d}.txt"), os.path.join("dogs_data","labels","train"))
    f.close()
    f = open("dogs_data/val.txt", "w")
    for d in val:
        f.write(f"./images/val/{d}.jpg\n")
        shutil.move(os.path.join("dogs_data","images", f"{d}.jpg"), os.path.join("dogs_data","images","val"))
        shutil.move(os.path.join("dogs_data", f"{d}.txt"), os.path.join("dogs_data","labels","val"))
    f.close()