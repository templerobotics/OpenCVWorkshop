
import cv2
import numpy as np
import os.path

#Paths to resources
loadpath = os.path.join('ColorHighlightResources', 'red_coat.jpg')
savepath = os.path.join('ColorHighlightResources', 'red_coat_alt.jpg')

#Define the range of HSV values to isolate / detect
lower_blue = np.array([115, 150, 50], dtype = np.uint8)
upper_blue = np.array([130, 255, 255], dtype = np.uint8)

#Load an image
img = cv2.imread(loadpath)


#To better control the color selection, switch to the HSV colorspace
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)



#Create a mask which represents areas of the image that fit our color range
img_bluemask = cv2.inRange(img_hsv, lower_blue, upper_blue)

#Create the negative of the mask, to "find" everywhere in the image that isn't
#in out color range
img_bluemask_not = cv2.bitwise_not(img_bluemask)



#Segment the image into our target color region and make the rest of the image
#black
img_blue = cv2.bitwise_and(img, img, mask = img_bluemask)

#Using the negative mask, black out the target colored section of the image
img_blue_not = cv2.bitwise_and(img, img, mask = img_bluemask_not)



#Make the "background" portion of the image grayscale, for visual effect
img_gray = cv2.cvtColor(img_blue_not, cv2.COLOR_BGR2GRAY)

#Grayscale images are only 1 channel, but we need a 3 channel image to merge
#with our colored portion. Numpy helps us extend the array
img_gray_3_channel = np.repeat(img_gray[:, :, np.newaxis], 3, axis = 2)



#Merge the colored section of the image with our grayscale background
img_grayblue = cv2.bitwise_or(img_blue, img_gray_3_channel)



#Save the image
cv2.imwrite(savepath, img_grayblue)
