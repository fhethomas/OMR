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
def pdf_to_img(file_path,destination_folder,img_name,split_img=False,split_dic={}):
    """convert pdf to image
    Args:
    file_path : pdf to be turned into images
    destination_folder : folder images are to be deposited
    img_name : name of image files you create - will be appended with a number
    split_img : optional argument: 'horizontal' or' 'vertical'
    split_dic : dictionary of the page numbers that you want - so if split two pages horizontally
                you will want {0:[7,1]} to split page 0 into image 7 and image 1"""
    pages = convert_from_path(file_path, 200,fmt="jpg")
    if split_img !=False and len(split_dic.keys())==0:
        split_dic = {count:[count*2+1,count*2+2] for count,p in enumerate(pages)}
    for count,page in enumerate(pages):
        if split_img==False:
            img_name = "{0}/{1}_{2}.jpg".format(destination_folder,img_name,count)
            page.save(img_name, 'JPEG')
        else:
            img_name_1 = "{0}/{1}_{2}.jpg".format(destination_folder,img_name,split_dic[count][0])
            img_name_2 = "{0}/{1}_{2}.jpg".format(destination_folder,img_name,split_dic[count][1])
            img1,img2 = image_splitter(page,split_img)
            img1.save(img_name_1,'JPEG')
            img2.save(img_name_2,'JPEG')
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    # change last arg from 1 to 0.1
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    return result

def image_splitter(img,dimension="horizontal"):
    """
    img : image loaded
    dimension: 'horizontal' or 'vertical'
    """
    imgwidth, imgheight = img.size
    if dimension.lower()=="horizontal":
        img1 = img.crop((0,0,int(imgwidth/2),imgheight))
        img2 = img.crop((int(imgwidth/2),0,imgwidth,imgheight))
    elif dimension.lower()=="vertical":
        img1 = img.crop((0,0,imgwidth,int(imgheight/2)))
        img2 = img.crop((0,int(imgheight/2),imgwidth,imgheight))
    else:
        print("dimension argument incorrect")
        return None
    return img1,img2





