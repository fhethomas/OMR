# OMR
Optical Mark Recognition Project

This project is specifically for converting scanned surveys into scores. 

The main way this works is by creating a dictionary of a clean (unfilled version) of the survey and a filled version of the survey and then comparing each answer box to see which has changed the most.

## Main functions

You'll need to run img_dictionary_creator functions on a clean version of the survey and a filled version to create 2 dictionaries to feed into score_return.

img_dictionary_creator(img_str,df,page=2,image_border = 5,clean=False,show_img=False)

img_dictionary_creator6_7(img_str,df,page=6,image_border = 5,clean=False)

score_return(return_filled_dic,return_clean_dic,show_img=True)



Both functions require an excel document to load a dataframe with question co-ordinates. Function 6_7 will take each answer as a variable, but original function took each question as a variable. 

Images will be returned if you want showing the filled answer.

![image](https://user-images.githubusercontent.com/29797377/145192852-e489e44b-4e75-48b4-9271-1a4efac87c07.png)

