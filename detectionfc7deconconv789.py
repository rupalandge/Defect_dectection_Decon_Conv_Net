# from keras import backend as K
from tensorflow.keras import backend as K
from tensorflow import keras as k
# from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
from keras.models import load_model
# from keras.preprocessing import image
from keras.optimizers import Adam
# from scipy.misc import imread
from matplotlib.pyplot import imread
import numpy as np
from matplotlib import pyplot as plt
import os
import numpy
import cv2

from models.FC7Decon1Con789 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder_defect.ssd_output_decoder import decode_detections
def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]
import glob

from PIL import Image
img_height = 512
img_width = 512
img_channels = 3

mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
n_classes =1
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05,2.0] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 0.05, 1.0] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
aspect_ratios = [[1.0, 2.0, 0.5],
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5],
    [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
steps = [8, 16, 32, 64, 128, 256, 512]
two_boxes_for_ar1 = True

offsets =[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

K.clear_session()
model = ssd_512(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales_coco ,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)


# 2: Load the trained VGG-16 weights into the model.

# TODO: Set the path to the VGG-16 weights.
#weights_path = 'G:/ssd_keras_1_master/vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5'
# weights_path = 'G:/ssd_keras_1_master/logs/New folder/ssd300_weights_epoch-35_loss-3.4457_val_loss-2.9737.h5'
## This weight is for the epoch 25 when Pc crahed
# weights_path = 'G:/SWIPENet_master2/Trainigweightconv7deconv/fc7cdecon1con789/ssd512_URPC2018_epoch-19.h5'
# weights_path = 'G:/SWIPENet_master2/Trainigweightconv7deconv/fc7cdecon1con789/again/ssd512fc7decon1conv789_weights.h5'
weights_path = 'D:/3/SWIPENet_master2/trainingfileFC7decon1con789_100epoch/weightfileFC7decon1conv789epoch30/ssd512Fc7decon1conv789_URPC2018_epoch-100.h5'
# weights_path = 'G:/SWIPENet_master2/trainingfileFC7decon1con789/ssd512fc7decon1conv789defect_weights.h5'

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=["accuracy"])




orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.


batch_holder = np.zeros((30, img_height, img_width, 3))

# img_dir =  "G:/ssd_keras_1_master/row image/"
img_dir =  "D:/3/SWIPENet_master2/testdataimage"

# img_dir =  "G:\ssd_keras_1_master\datasets1\VOCdevkit\VOC2007\JPEGImages/"
for i,img, in enumerate(os.listdir(img_dir)):
    img = os.path.join(img_dir, img)
    orig_images.append(imread(img))
    img =image.load_img(img, target_size=(img_height, img_width))
    # img = image.load_img(os.path.join(img_dir,img), target_size=(img_height, img_width ))
    batch_holder[i,:] =img


    y_pred = model.predict(batch_holder)
    print(y_pred)

    Roc = roc_curve(y_pred)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[i])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    classes = ['background',
               'defect']

    plt.figure(figsize=(5,5))
    # plt.imshow(orig_images[i])

    current_axis = plt.gca()
    for box in y_pred_thresh[i]:
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








# #plot Perfomance
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['acc','val_acc'], loc= 'lower right')
# plt.show()



# orig_images = [] # Store the images here.
# input_images = [] # Store resized versions of the images here.
#
#
# batch_holder = np.zeros((30, img_height, img_width, 3))
#
# # img_dir =  "G:/ssd_keras_1_master/row image/"
# img_dir =  "G:\ssd_keras_1_master\datasets1\VOCdevkit\VOC2007\JPEGImages/"
# for i,img, in enumerate(os.listdir(img_dir)):
#     img = os.path.join(img_dir, img)
#     orig_images.append(imread(img))
#     img =image.load_img(img, target_size=(img_height, img_width))
#     # img = image.load_img(os.path.join(img_dir,img), target_size=(img_height, img_width ))
#     batch_holder[i,:] =img
#
#     y_pred = model.predict(batch_holder)
#     print(y_pred)
#
#     y_pred_decoded = decode_detections(y_pred,
#                               confidence_thresh=0.5,
#                               iou_threshold=0.45,
#                               top_k=200,
#                               input_coords='centroids',
#                               normalize_coords=True,
#                               img_height=img_height,
#                               img_width=img_width)
#
#     np.set_printoptions(precision=2, suppress=True, linewidth=90)
#     print("Predicted boxes:\n")
#     print('    class    conf  xmin     ymin   xmax    ymax')
#     print(y_pred_decoded[i])
#
#     # Display the image and draw the predicted boxes onto it.
#
#     # Set the colors for the bounding boxes
#     colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#
#     classes = ['background',
#                'defect']
#
#     plt.figure(figsize=(20, 12))
#     # plt.imshow(orig_images[i])
#
#     current_axis = plt.gca()
#     for box in y_pred_decoded[i]:
#         xmin = box[-4] * orig_images[0].shape[1] / img_width
#         ymin = box[-3] * orig_images[0].shape[0] / img_height
#         xmax = box[-2] * orig_images[0].shape[1] / img_width
#         ymax = box[-1] * orig_images[0].shape[0] / img_height
#         color = colors[int(box[0])]
#         label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#         current_axis.add_patch(
#         plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
#         current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
#         # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#     plt.imshow(orig_images[i])
#     plt.show()
