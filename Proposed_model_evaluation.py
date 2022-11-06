from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.conv7deconv import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detectiondefect_2d_data_generator import DataGenerator
from pascal_voc_utils import predict_all_to_txt
from eval_utils.average_precision_evaluatorProposed import Evaluator

# Set the input image size for the model.
img_height = 512
img_width = 512
img_channels = 3
detection_mode = 'test'
model_mode = 'inference'
# 1: Build the Keras model
mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
# n_classes =7
n_classes =1
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
                mode='training',
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

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'G:/SWIPENet_master2/Trainigweightconv7deconv/ssd512_URPC2018_epoch-10.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Or load trained model
# TODO: Set the path to the `.h5` file of the model to be loaded.
# model_path = 'path/to/trained/model.h5'
#
# # We need to create an SSDLoss object in order to pass that to the model loader.
# ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
#
# K.clear_session() # Clear previous models from memory.
#
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})



# TODO: Set the paths to the dataset here.
dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir ='G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = 'G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'
Pascal_VOC_dataset_image_set_filename ='G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
classes = ['background','defect']
dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

batch_size = 8# Ideally, choose a batch size that divides the number of images in the dataset.

predict_all_to_txt(model=model,
                   img_height=img_height,
                   img_width=img_width,
                   batch_generator=dataset,
                   batch_size=batch_size,
                   batch_generator_mode='resize',
                   classes=['background',
                            'defect'],
                   out_file_prefix='ssd300_07+12_2007_test_eval/comp3_det_test_',
                   confidence_thresh=0.01,
                   iou_threshold=0.45,
                   top_k=200,
                   pred_coords='centroids',
                   normalize_coords=True)


evaluator = Evaluator(model=model,
                          n_classes=n_classes,
                          data_generator=dataset,
                          model_mode=model_mode,
                          detection_mode=detection_mode)
results = evaluator(img_height=img_height,
                        img_width=img_width,
                        batch_size=32,
                        data_generator_mode='resize',
                        round_confidences=False,
                        matching_iou_threshold=0.5,
                        border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample',
                        num_recall_points=11,
                        ignore_neutral_boxes=True,
                        return_precisions=True,
                        return_recalls=True,
                        return_average_precisions=True,
                        verbose=True)

print('Detection results of multiple models have been saved in /data/deeplearn/SWEIPENet/dataset/Detections/...')