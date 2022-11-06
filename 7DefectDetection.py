from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt
import os
from models.FC7Decon1Con789 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detectiondefect_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 512
img_width = 512
img_channels = 3

mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
# n_classes = 7 ## This is for 7type defect
n_classes = 6
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
#
# weights_path = 'G:/SWIPENet_master2/Trainigweightconv7deconv/fc7cdecon1con789/ssd512_URPC2018_epoch-19.h5'
# weights_path = 'G:/SWIPENet_master2/6typeDefect150EpochWeight/ssd512_6DEFECTdataset_epoch-25.h5'
weights_path = 'G:/SWIPENet_master2/ssd512_6defect_2fc7decon1conv789_weights.h5' ##this weight for 7defect
# weights_path = 'G:/SWIPENet_master2/6typedefectweight/ssd51_6defect_2fc7decon1conv789_weights.h5'

model.load_weights(weights_path, by_name=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=["accuracy"])


orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.
batch_holder = np.zeros((30, img_height, img_width, 3))



#
# img_dir =  "G:/ssd_keras_1_master/row image/"
img_dir =  "G:/NEU-DET-Steel-Surface-Defect-Detection-master/NEU-DET-Steel-Surface-Defect-Detection-master/Validation_Images/"
for i,img, in enumerate(os.listdir(img_dir)):
    img = os.path.join(img_dir, img)
    orig_images.append(imread(img))
    img =image.load_img(img, target_size=(img_height, img_width))
    # img = image.load_img(os.path.join(img_dir,img), target_size=(img_height, img_width ))
    batch_holder[i,:] =img


    y_pred = model.predict(batch_holder)
    print(y_pred)

    confidence_threshold = 0.24

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[i])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    classes = ['background', 'rolled-in_scale', 'scratches', 'pitted_surface', 'patches', 'inclusion', 'crazing']
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

    output_dir ="G:/SWIPENet_master2/6typeDefectSavefile/"
    plt.imshow(orig_images[i])
    plt.show()
    fnum = int(100)
    plt.savefig(f"{fnum:04d}.png")
    # plt.savefig('{}/graph.png'.format(output_dir))


# # This code for decode detection ##
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
#     colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#     classes = ['background',
#                'defect']
#     plt.figure(figsize=(20, 12))
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
#     plt.imshow(orig_images[i])
#     plt.show()
# #
# #
#


#
# dataset = DataGenerator()
#
# # TODO: Set the paths to the datasets here.
#
# VOC_2007_images_dir         = 'G:/SWIPENet_master2/6typeTestDefectDataset/JPEG/'
# VOC_2007_annotations_dir    =  'G:/SWIPENet_master2/6typeTestDefectDataset/ANNOTATIONS/'
# VOC_2007_test_image_set_filename = 'G:/SWIPENet_master2/6typeTestDefectDataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
#
# # The XML parser needs to now what object class names to look for and in which order to map them to integers.
# classes = ['background', 'rolled-in_scale', 'scratches', 'pitted_surface', 'patches', 'inclusion', 'crazing']
#
# dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
#                   image_set_filenames=[VOC_2007_test_image_set_filename],
#                   annotations_dirs=[VOC_2007_annotations_dir],
#                   classes=classes,
#                   include_classes='all',
#                   exclude_truncated=False,
#                   exclude_difficult=True,
#                   ret=False)
#
# convert_to_3_channels = ConvertTo3Channels()
# resize = Resize(height=img_height, width=img_width)
#
# generator = dataset.generate(batch_size=1,
#                              shuffle=True,
#                              transformations=[convert_to_3_channels,
#                                               resize],
#                              returns={'processed_images',
#                                       'filenames',
#                                       'inverse_transform',
#                                       'original_images',
#                                       'original_labels'},
#                              keep_images_without_gt=False)
#
# # Generate a batch and make predictions.
#
# batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator)
#
# i = 0 # Which batch item to look at
#
# print("Image:", batch_filenames[i])
# print()
# print("Ground truth boxes:\n")
# print(np.array(batch_original_labels[i]))
#
#
# y_pred = model.predict(batch_images)
#
#
# confidence_threshold = 0.5
#
# # Perform confidence thresholding.
# y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
#
# # Convert the predictions for the original image.
# y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)
#
# np.set_printoptions(precision=2, suppress=True, linewidth=90)
# print("Predicted boxes:\n")
# print('   class   conf xmin   ymin   xmax   ymax')
# print(y_pred_thresh_inv[i])
# # Display the image and draw the predicted boxes onto it.
#
# # Set the colors for the bounding boxes
# colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
#
# plt.figure(figsize=(20,12))
# plt.imshow(batch_original_images[i])
#
# current_axis = plt.gca()
#
# for box in batch_original_labels[i]:
#     xmin = box[1]
#     ymin = box[2]
#     xmax = box[3]
#     ymax = box[4]
#     label = '{}'.format(classes[int(box[0])])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
#
# for box in y_pred_thresh_inv[i]:
#     xmin = box[2]
#     ymin = box[3]
#     xmax = box[4]
#     ymax = box[5]
#     color = colors[int(box[0])]
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
#
# plt.show()