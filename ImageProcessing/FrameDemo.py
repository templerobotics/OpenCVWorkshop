
import cv2
import numpy as np
import os.path

loadpath_subject = os.path.join('FrameResources', 'willy.jpg')
loadpath_frame = os.path.join('FrameResources', 'frame.png')
savepath = os.path.join('FrameResources', 'merged.png')

frame = cv2.imread(loadpath_frame)

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.imwrite('frame_gray.png', frame_gray)

_, frame_mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY)

frame_mask_negative = cv2.bitwise_not(frame_mask)

subject = cv2.imread(loadpath_subject)

subject_scaled = cv2.resize(subject, (frame.shape[1], frame.shape[0]))

subject_cutout = cv2.bitwise_and(subject_scaled, subject_scaled, mask = frame_mask_negative)

merged = cv2.bitwise_or(frame, subject_cutout)

cv2.imwrite(savepath, merged)
