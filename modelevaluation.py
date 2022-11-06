from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.conv7deconv import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detectiondefect_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator


img_height = 512
img_width = 512
img_channels = 3
model_mode = 'inference'

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
                mode=model_mode,
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
#
weights_path = 'G:/SWIPENet_master2/Trainigweightconv7deconv/ssd512_URPC2018_epoch-10.h5'
model.load_weights(weights_path, by_name=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# # TODO: Set the path to the `.h5` file of the model to be loaded.
# model_path = 'path/to/trained/model.h5'
#
# # We need to create an SSDLoss object in order to pass that to the model loader.
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
#
# K.clear_session() # Clear previous models from memory.
#
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'DecodeDetections': DecodeDetections,
#                                                'compute_loss': ssd_loss.compute_loss})

dataset = DataGenerator()
VOC_2013_images_dir = 'G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/JPEGImages/'
VOC_2013_annotations_dir = 'G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'
VOC_2013_trainval_image_set_filename = 'G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2013_test_image_set_filename = 'G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

classes = ['background','defect']
dataset = DataGenerator()
dataset.parse_xml(images_dirs=[VOC_2013_images_dir],
                        image_set_filenames=[VOC_2013_test_image_set_filename],
                        # sample_weights_dirs=VOC_2013_sampleweights_dir,
                        annotations_dirs=[VOC_2013_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)
#
# # TODO: Set the paths to the dataset here.
# dataset = DataGenerator()
#
# # TODO: Set the paths to the dataset here.
# Pascal_VOC_dataset_images_dir ='G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/JPEGImages/'
# Pascal_VOC_dataset_annotations_dir = 'G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'
# Pascal_VOC_dataset_image_set_filename ='G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
# classes = ['background','defect']
# dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
#                   image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
#                   annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
#                   classes=classes,
#                   include_classes='all',
#                   exclude_truncated=False,
#                   exclude_difficult=False,
#                   ret=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
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

mean_average_precision, average_precisions, precisions, recalls = results


for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes: break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0,1,11))
        cells[i, j].set_yticks(np.linspace(0,1,11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)

plt.show()
evaluator.get_num_gt_per_class(ignore_neutral_boxes=True,
                               verbose=False,
                               ret=False)

evaluator.match_predictions(ignore_neutral_boxes=True,
                            matching_iou_threshold=0.5,
                            border_pixels='include',
                            sorting_algorithm='quicksort',
                            verbose=True,
                            ret=False)

precisions, recalls = evaluator.compute_precision_recall(verbose=True, ret=True)

average_precisions = evaluator.compute_average_precisions(mode='integrate',
                                                          num_recall_points=11,
                                                          verbose=True,
                                                          ret=True)

mean_average_precision = evaluator.compute_mean_average_precision(ret=True)


for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))
# batch_size = 8# Ideally, choose a batch size that divides the number of images in the dataset.