#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pdf2image import convert_from_path
import os
from PIL import Image
# compare two images
# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.optimize import minimize
import math
import pandas as pd


# In[29]:


def jpg_to_pdf(folder_path,destination_folder,file_name,rotate_angle=False):
    os.chdir(folder_path)
    im_list = []
    pdf1_filename = "{0}/{1}.pdf".format(destination_folder,file_name)
    img_one = True
    file_list = sorted(os.listdir())
    for file in file_list:
        if file[-3:]=="jpg":
            if img_one==True:
                im1 = Image.open("{0}/{1}".format(folder_path,file))
                if rotate_angle!=False:
                    im1  = rotation_img(im1,rotate_angle)
                img_one=False
            else:
                im = Image.open("{0}/{1}".format(folder_path,file))
                if rotate_angle!=False:
                    im  = rotation_img(im,rotate_angle)
                im_list.append(im)
    im1.save(pdf1_filename, "PDF" ,resolution=100.0, save_all=True, append_images=im_list)
def rotation_img(img,rotate_angle):
    if rotate_angle in [90,180,270]:
        if rotate_angle==90:
            return img.transpose(Image.ROTATE_90)
        elif rotate_angle==180:
            return img.transpose(Image.ROTATE_180)
        elif rotate_angle==270:
            return img.transpose(Image.ROTATE_270)
        else:
            print("error")
            return None
    else:
        return img.rotate(rotate_angle,fillcolor=(255, 255, 255))
def img_grey(img_str):
    target_img = cv2.imread(img_str)
    gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    return gray_img
def pdf_to_img(file_path,destination_folder):
    # convert pdf to image
    pages = convert_from_path(file_path, 200,fmt="jpg")
    for count,page in enumerate(pages):
        img_name = "{0}/page_{1}.jpg".format(destination_folder,count)
        page.save(img_name, 'JPEG')
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    # change last arg from 1 to 0.1
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    return result


# In[ ]:




