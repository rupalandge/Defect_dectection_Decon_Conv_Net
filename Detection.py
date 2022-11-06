from tensorflow.keras import backend as K
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from matplotlib.pyplot import imread
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
import os
import numpy
import cv2

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from matplotlib import pyplot as plt
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize



img_height = 512
img_width = 512
img_channels = 3
mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
n_classes =7
# n_classes =1
scales_pascal = [0.3, 0.15, 0.07, 0.04]
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0]]
two_boxes_for_ar1 = True
steps = [16, 8, 4]
offsets = [0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

K.clear_session()
model = ssd_512(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

weights_path = 'D:/3/SWIPENet_master2/trainingfileFC7decon1con789_100epoch/ssd512fc7decon1conv789defect.h5'
model.load_weights(weights_path, by_name=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss,metrics=["accuracy"])



orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.


batch_holder = np.zeros((10, img_height, img_width, 3))

# img_dir =  "G:/ssd_keras_1_master/row image/"
img_dir =  "D:/3/SWIPENet_master2/image/"

for i,img, in enumerate(os.listdir(img_dir)):

    img = os.path.join(img_dir, img)

    orig_images.append(imread(img))

    img =image.load_img(img, target_size=(img_height, img_width))

    # img = image.load_img(os.path.join(img_dir,img), target_size=(img_height, img_width ))
    batch_holder[i,:] = img

    y_pred = model.predict(batch_holder)
    print("this is the predicted value",y_pred)

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('    class    conf  xmin     ymin   xmax    ymax')
    print(y_pred_decoded[i])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    classes = ['background', 'seacucumber', 'seaurchin', 'scallop', 'starfish', 'holothurian', 'echinus', 'waterweeds']

    plt.figure(figsize=(20, 12))
    # plt.imshow(orig_images[i])

    current_axis = plt.gca()

    # for box in y_true[i]:
    #     xmin = box[1]
    #     ymin = box[2]
    #     xmax = box[3]
    #     ymax = box[4]
    #     label = '{}'.format(classes[int(box[0])])
    #     current_axis.add_patch(
    #         plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
    #     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': 'green', 'alpha': 1.0})

    for box in y_pred_decoded[i]:
        xmin = box[-4] * orig_images[0].shape[1] / img_width
        ymin = box[-3] * orig_images[0].shape[0] / img_height
        xmax = box[-2] * orig_images[0].shape[1] / img_width
        ymax = box[-1] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    plt.imshow(orig_images[i])

    plt.show()

# #
# input_image = []
# import os
# from skimage import io
# from skimage.color import rgb2gray
# from skimage.transform import resize
# import numpy
# import scipy.ndimage
#
# import numpy as np
# from PIL import Image
#
#
# imgage= []
# imgski = numpy.zeros((6, 800, 800, 1), )
# img = numpy.zeros((6, 28, 28, 1), dtype=numpy.uint8)
# new_array = numpy.zeros((img.shape[0], 28, 28,1), dtype=numpy.float32)
# img_res = numpy.zeros((6, 28, 28, 1), dtype=numpy.uint8)
# for i,img, in enumerate((os.listdir(img_dir))):
#     img = os.path.join(img_dir,img )
#     # img = cv2.imread(img)
#     # gray = rgb2gray(img)
#     # gray= gray[...,np.newaxis]
#     # print(gray.shape[2])
#     # # gray =np.expand_dims(gray, axis=2)
#     # imgski[i]= gray
#     # print(imgski[1])
#     # print(gray)
#     img = Image.open(img)
#     imgage.append(img)
#     img =img.convert("L")
#     img = img.resize((28, 28),Image.ANTIALIAS)
#     img = np.array(img)
#     img = img.reshape(28,28,1)
#     arr = numpy.array(img, dtype=numpy.uint8)
#     new_array[i] = arr
# print(new_array)
