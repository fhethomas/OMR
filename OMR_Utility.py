#!/usr/bin/env python
# coding: utf-8

# In[1]:

# IMPORT Section

print("Imports started...")
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
from sklearn.cluster import KMeans
import keras_ocr
import imagehash
print("Imports loaded")


# Utility Functions
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver
def rectContour(contours):
    # find rectangular contours
    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        
        if area > 50:
            peri = cv2.arcLength(i, True)
            
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon
def getCornerPoints(cont):
    # get the corner points of the contours
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx
def reorder(myPoints):
    # reorders the points
    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew
def find_nearest(array, value):
    array = np.asarray(array[:,1])
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def img_grey(img_str):
    target_img = cv2.imread(img_str)
    gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    return gray_img
def cont_func(img):
    #imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)
    contours, heirarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    rectangle_contours = rectContour(contours)
    return rectangle_contours
def show_img_coords(img,coords):
    corners = cv2.boundingRect(coords)
    x = corners[0]
    y = corners[1]
    w = corners[2]
    h = corners[3]
    sub_img = img[y:y+h, x:x+w]
    plt.imshow(sub_img)
    plt.show()
def return_img_coords(coords):
    corners = cv2.boundingRect(coords)
    x = corners[0]
    y = corners[1]
    w = corners[2]
    h = corners[3]
    return x,y,w,h
def key_create(arr):
    return "_".join([str(x) for x in list(arr.flatten())])
def key_split(dic):
    arr=[x.split("_") for x in list(dic.keys())]
    #print(len(arr))
    arr = np.array(arr)
    arr = arr.astype(np.int)
    return arr
def find_comparison_img(test_img_dic,example_img_dic):
    test_img_coords = key_split(test_img_dic)
    example_img_coords = key_split(example_img_dic)
    if test_img_coords.size != example_img_coords.size:
        print("Questions or Answers Missing")
        
    nearest_co_ords = {}
    img_dic = {}
    for k in test_img_coords:
        proximity_arr = np.square(example_img_coords-k)
        #proximity_arr[:,1]=proximity_arr[:,1]/2
        proximity_arr=np.sum(proximity_arr,axis=1)
        nearest = np.min(proximity_arr)
        #print(k)
        #print(proximity_arr)
        #print(nearest)
        #print(example_img_coords[proximity_arr==nearest,:])
        #print(example_img_coords)
        
        # create a dictionary of the array and the nearest element
        created_k = key_create(k)
        nearest_co_ords[created_k] = example_img_coords[proximity_arr==nearest,:]
        # create dictionary of images
        img_dic[created_k]=[test_img_dic[created_k],example_img_dic[key_create(nearest_co_ords[created_k])]]
    #print(nearest_co_ords)
    return nearest_co_ords,img_dic
# can we use image hashing to compare values of 2 images


# tryng to compare images - working on using image hashing maybe - not 100% on if this will work
def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    #return str(sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v]))
    
    # just going to use image hash library
    return imagehash.whash(Image.fromarray(image))
    
def hamming_distance(string1, string2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance
#ssim(test_imgs[0],test_imgs[1])
"""
OLD FUNCTION BELOW USED SSIM - I found this didn't work.
Best thing to do was just grey scale and find % filled and compare to the original

def compare_img(img1,img2,cvt_grey=False):
    if cvt_grey==True:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1,w1=img1.shape
    dim = (w1,h1)
    img2 = cv2.resize(img2,dim)
    h2,w2=img2.shape
    lowest_h = min(h1,h2)-5
    lowest_w = min(w1,w2)-5
    img1 = img1[5:lowest_h,5:lowest_w]
    img2 = img2[5:lowest_h,5:lowest_w]
    return ssim(img1,img2)"""
def img_filled_percent(img):
    img = img.flatten()
    total = img.size
    filled = img[img!=0].size
    return round(filled/total,4)
def compare_img(img1,img2,cvt_grey=False):
    if cvt_grey==True:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1,w1=img1.shape
    dim = (w1,h1)
    img2 = cv2.resize(img2,dim)
    h2,w2=img2.shape
    lowest_h = min(h1,h2)-5
    lowest_w = min(w1,w2)-5
    img1 = img1[5:lowest_h,5:lowest_w]
    img2 = img2[5:lowest_h,5:lowest_w]
    
    img1_percent = img_filled_percent(img1)
    img2_percent = img_filled_percent(img2)
    #print("filled: {0}, clean: {1}".format(img1_percent,img2_percent))
    return img1_percent-img2_percent
def find_best_answer(img_dic,questions=9):
    # find the clusters of questions by height - get centroids
    kmeans_model = KMeans(n_clusters=9)
    t1=key_split(img_dic)
    t1=t1[np.argsort(t1[:, 1])]
    kmeans_model.fit(t1[:,1].reshape(-1,1))
    centroids = np.sort(kmeans_model.cluster_centers_.flatten())
    question = {centroids[x]:x+1 for x in range(centroids.size)}
    question_assignment = {}
    # assign each image to a question
    for t in t1:
        height_var = t[1]
        distance = np.abs(centroids - height_var)
        min_distance = min(distance)
        closest = centroids[distance==min_distance][0]

        k_var = question[closest]
        if k_var in question_assignment.keys():
            question_assignment[k_var].append(t)
        else:
            question_assignment[k_var] = [t]
    final_result_dic = {}
    count = 0
    for q in question_assignment.keys():
        print(count)
        count+=1
        #print(q)
        result_dic = {}
        for k in question_assignment[q]:
            #print(k)
            img_comp = img_dic[key_create(k)]
            # Some images just error - so have built in to ignore - just look at questions we can answers
            try:
                ssim_score, hash_score = compare_img(img_comp[0],img_comp[1])
            except:
                ssim_score, hash_score = 1, 10
            #print(ssim_score)
            result_dic[ssim_score]=[img_comp[0]]
        res_arr = np.sort(list(result_dic.keys()))
        try:
            final_result_dic[q]= result_dic[res_arr[0]]
        except:
            return final_result_dic
    return final_result_dic

"""# CURRENTLY GETTING RID OF THIS WORK AROUND
def score_return(return_filled_dic,return_clean_dic,show_img=True):
    q_a = []
    for k in return_filled_dic.keys():
        comparison_list=[]
        # Filled 2nd biggest box is actually empty - need a work around
        if k[2]!="1":
            # Currently correcting the key to 1 from our example image
            if k[2]=="2":
                clean_key = k[:2] + "1" + k[3:]
            else:
                clean_key=k
            print(clean_key)
            for i in range(len(return_filled_dic[k])):
                t = compare_img(return_filled_dic[k][i],return_clean_dic[clean_key][i])
                comparison_list.append(t)
            sorted_comparison_list = sorted(comparison_list)
            comp_in = comparison_list.index(max(comparison_list))
            first_largest_score = sorted_comparison_list[-1]
            first_index =  comparison_list.index(first_largest_score)
            second_largest_score = sorted_comparison_list[-2]
            second_index = comparison_list.index(second_largest_score)
            if show_img==True:
                plt.imshow(return_filled_dic[k][comp_in])
                print("Index returned: {0}".format(comp_in))
                plt.show()
            q_a.append([clean_key,first_index,first_largest_score,second_index,second_largest_score])
    return q_a"""
def score_return(return_filled_dic,return_clean_dic,show_img=True):
    q_a = []
    for k in return_filled_dic.keys():
        comparison_list=[]
        # Filled 2nd biggest box is actually empty - need a work around

        for i in range(len(return_filled_dic[k])):
            t = compare_img(return_filled_dic[k][i],return_clean_dic[k][i])
            comparison_list.append(t)
        sorted_comparison_list = sorted(comparison_list)
        comp_in = comparison_list.index(max(comparison_list))
        first_largest_score = sorted_comparison_list[-1]
        first_index =  comparison_list.index(first_largest_score)
        second_largest_score = sorted_comparison_list[-2]
        second_index = comparison_list.index(second_largest_score)
        if show_img==True:
            plt.imshow(return_filled_dic[k][comp_in])
            print("Index returned: {0}".format(comp_in))
            plt.show()
        q_a.append([k,first_index,first_largest_score,second_index,second_largest_score])
    return q_a
def img_dictionary_creator(img_str,df,page=2,image_border = 5,clean=False,show_img=False):
    """
    Output : a dictionary of questions & Answers: {Page_Box_Question : [Image1,Image2]}
    """
    return_dictionary = {}
    threshhold_level = 180
    img = cv2.imread(img_str)
    imgContours = img.copy()
    imgBiggestContour = img.copy()
    # grey scale
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)
    boxes = max(df[df["Page"]==page]["Box"])
    # find contours of the page
    contours, heirarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # find rectangles
    rectangle_contours = rectContour(contours)
    if clean==True:
        boxes-=1
    for box in range(0,boxes+1):
        biggestCorner = reorder(getCornerPoints(rectangle_contours[box]))
        x,y,w,h=return_img_coords(biggestCorner)
        imgWarpColoured = img[y:y+h, x:x+w]
        img_small_cut = img[y:y+h, x:x+w]
        # Uncomment below to see each box
        #plt.imshow(img_small_cut)
        #plt.show()
        # Apply threshold so that we can look at a binary response to pixels

        # Commented out - as may need coloured image to find best squares
        imgWarpGrey = cv2.cvtColor(imgWarpColoured,cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGrey,threshhold_level,255,cv2.THRESH_BINARY_INV)[1]
        imgBiggestContour = imgThresh.copy()

        # proportion that question consumes:
        # Get rid of the question side of the box - so we can just look at the answers
        question_percentage = min(df[(df["Box"]==box)&(df["Page"]==page)]["Question Right Percentage"])
        h,w=imgBiggestContour.shape
        left = int(w*question_percentage)
        imgBiggestContour=imgBiggestContour[:,left:w]
        questions = max(df[(df["Box"]==box)&(df["Page"]==page)]["Question"])
        cummulative_percentage = 0

        # iterate over the questions
        for q in range(questions):
            return_key = "{0}_{1}_{2}".format(str(page),str(box),str(q))
            # creates a question box based on height
            height_interval_percentage = list(df[(df["Box"]==box)&(df["Page"]==page)&(df["Question"]==q+1)]["PercentagePageHeight"])
            height_interval_percentage=height_interval_percentage[0]
            answer_number = list(df[(df["Box"]==box)&(df["Page"]==page)&(df["Question"]==q+1)]["Answer Number"])
            answer_number = answer_number[0]
            top_tick = int(cummulative_percentage*h)
            bottom_tick = int(cummulative_percentage*h + height_interval_percentage * h)
            img_row = imgBiggestContour[top_tick+image_border:bottom_tick-image_border,:]
            img_height, img_width = img_row.shape
            # Uncomment below to show each question row
            #plt.imshow(img_row)
            #plt.show()
            cummulative_width_percentage = 0
            # iterate over the answers - split them into sub images
            for a in range(answer_number):
                width_interval_percentage = list(df[(df["Box"]==box)&(df["Page"]==page)&(df["Question"]==q+1)]["A{0}".format(a+1)])
                width_interval_percentage = width_interval_percentage[0]
                left_tick = int(cummulative_width_percentage*img_width)
                
                #print("Box: {0}, Question: {1}, Answer: {2}, cummulative_width_percentage: {3}, width_interval_percentage: {4}, img_width: {5}".format(str(box),str(q),str(a),str(cummulative_width_percentage),str(width_interval_percentage),str(img_width)))
                right_tick = int(cummulative_width_percentage*img_width + width_interval_percentage * img_width)
                answer_img = img_row[:,left_tick+image_border:right_tick-image_border]
                cummulative_width_percentage+=width_interval_percentage
                # Add the answer image to the dictionary
                if return_key in return_dictionary.keys():
                    return_dictionary[return_key].append(answer_img)
                else:
                    return_dictionary[return_key] = [answer_img]
                if show_img==True:
                    print("Question: {0}".format(str(q)))
                    plt.imshow(answer_img)
                    plt.show()
            cummulative_percentage+=height_interval_percentage
    return return_dictionary

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
def image_splitter(img,dimension="horizontal"):
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
def img_dictionary_creator6_7(img_str,df,page=6,image_border = 5,clean=False):
    """
    Output : a dictionary of questions & Answers: {Page_Box_Question : [Image1,Image2]}
    """
    select_list = ["PercentageHeightfromBottom","PercentageHeighttoBottom","PercentagefromRight","PercentagetoRight"]
    return_dictionary = {}
    threshhold_level = 180
    img = cv2.imread(img_str)
    imgContours = img.copy()
    imgBiggestContour = img.copy()
    # grey scale
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)
    boxes = max(df[df["Page"]==page]["Box"])
    # find contours of the page
    contours, heirarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # find rectangles
    rectangle_contours = rectContour(contours)
    if clean==True:
        boxes-=1
    for box in range(0,boxes+1):
        biggestCorner = reorder(getCornerPoints(rectangle_contours[box]))
        x,y,w,h=return_img_coords(biggestCorner)
        imgWarpColoured = img[y:y+h, x:x+w]
        img_small_cut = img[y:y+h, x:x+w]
        # Uncomment below to see each box
        #plt.imshow(img_small_cut)
        #plt.show()
        # Apply threshold so that we can look at a binary response to pixels

        # Commented out - as may need coloured image to find best squares
        imgWarpGrey = cv2.cvtColor(imgWarpColoured,cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGrey,threshhold_level,255,cv2.THRESH_BINARY_INV)[1]
        imgBiggestContour = imgThresh.copy()

        # proportion that question consumes:
        # Get rid of the question side of the box - so we can just look at the answers
        questions = max(list(df[(df["Box"]==box)&(df["Page"]==page)]["Question"]))
        
        # iterate over the questions
        for q in range(1,questions+1):
            h,w=imgBiggestContour.shape
            #print("Question: {0}".format(str(q)))
            answer_df=df[(df["Box"]==box)&(df["Page"]==page)&(df["Question"]==q)]
            return_key = "{0}_{1}_{2}".format(str(page),str(box),str(q))
            
            # need to cut original image each time by the question &, then by the left/right/ up/down %
            
            question_percentage = max(answer_df["Question Right Percentage"])
            # cut the left side of the image off where the question is
            left = int(w*question_percentage)
            imgBiggestContour_Question=imgBiggestContour[:,left:w]
            h,w=imgBiggestContour_Question.shape
            answers = list(answer_df["Answer Number"])
            
            #iterate over the answers
            for a in answers:
                #print("Answer: {0}".format(str(a)))
                test_df = answer_df[answer_df["Answer Number"]==a][select_list]
                #["PercentageHeightfromBottom","PercentageHeighttoBottom","PercentagefromRight","PercentagetoRight"]
                # get the dimensions of the box you want
                pcHfB=test_df.iloc[0][0]
                pcHtB=test_df.iloc[0][1]
                pcfR=test_df.iloc[0][2]
                pctR=test_df.iloc[0][3]
                top,bottom,left,right = int(h*pcHfB)+image_border,int(h*pcHtB)-image_border,int(w*pcfR)+image_border,int(w*pctR)-image_border
                # img dimensions go height, width
                # return your image
                answer_img = imgBiggestContour_Question[top:bottom,left:right]
                #plt.imshow(answer_img)
                #plt.show()
                if return_key in return_dictionary.keys():
                    return_dictionary[return_key].append(answer_img)
                else:
                    return_dictionary[return_key] = [answer_img]
                #return test_df
    return return_dictionary
           
print("Functions built")
