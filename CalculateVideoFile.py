import cv2
import numpy as np

# filename = 'Path to the  Video file'
# file_size = (1200*1200)
#
# output_file = 'ssd_model_detection'
# output_frame_per_second = 20
#
# Resized_dimension = (512*512)
# img_norm_ratio =  0.007843
#
# neural_network =cv2.dnn.readNetFromCaffe()

cap = cv2.VideoCapture(0)

while True:
    sucess, img = cap.read()
    cv2.imshow("webcam_window", img)
    if cv2.waitKey(1) == ord('q'):
        break
