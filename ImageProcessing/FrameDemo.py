
import cv2
import numpy as np
import os.path

loadpath_subject = os.path.join('FrameResources', 'willy.jpg')
loadpath_frame = os.path.join('FrameResources', 'frame.png')
savepath = os.path.join('FrameResources', 'merged.png')

#Load the frame image
frame = cv2.imread(loadpath_frame)

#Convert the frame to grayscale so that it has a single channel
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#In the grayscale frame, create a mask which is white wherever there was color in the frame image
_, frame_mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY)

#Invert the mask so that the white area represents the empty part of the frame
frame_mask_negative = cv2.bitwise_not(frame_mask)

#Load the subject that we want to put in the frame
subject = cv2.imread(loadpath_subject)

#Scale the subject to the same size as the frame. Some warping will be visible
subject_scaled = cv2.resize(subject, (frame.shape[1], frame.shape[0]))

#Using the inverted mask, trim the scaled subject into the shape of the inside of the frame
subject_cutout = cv2.bitwise_and(subject_scaled, subject_scaled, mask = frame_mask_negative)

#Combine the frame and the cutout of the subject into a new, merged image
merged = cv2.bitwise_or(frame, subject_cutout)

#Save the merged image
cv2.imwrite(savepath, merged)
