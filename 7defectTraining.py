from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from math import ceil
import numpy as np
from models.FC7Decon1Con789 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder_defect.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detectiondefect_2d_data_generator import DataGenerator

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from matplotlib import pyplot as plt
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
import time
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 512
img_width = 512
img_channels = 3

mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
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
                mode='training',
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

weights_path = 'D:/3/ssd_keras_1_master/vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5'
model.load_weights(weights_path, by_name=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss,metrics=['accuracy'])


VOC_2013_images_dir = 'D:/3/NEU-DET-Steel-Surface-Defect-Detection-master/NEU-DET-Steel-Surface-Defect-Detection-master/IMAGES/'
VOC_2013_annotations_dir = 'D:/3/NEU-DET-Steel-Surface-Defect-Detection-master/NEU-DET-Steel-Surface-Defect-Detection-master/ANNOTATIONS/'
VOC_2013_trainval_image_set_filename = 'D:/3/NEU-DET-Steel-Surface-Defect-Detection-master/NEU-DET-Steel-Surface-Defect-Detection-master/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2013_test_image_set_filename = 'D:/3/NEU-DET-Steel-Surface-Defect-Detection-master/NEU-DET-Steel-Surface-Defect-Detection-master/VOCdevkit/VOC2007/ImageSets/Main/test.txt'


classes = ['background','rolled-in_scale','scratches','pitted_surface','patches','inclusion','crazing']
train_dataset = DataGenerator(load_images_into_memory= True, hdf5_dataset_path=None)

val_dataset = DataGenerator(load_images_into_memory= True , hdf5_dataset_path=None)

train_dataset.parse_xml(images_dirs=[VOC_2013_images_dir],
                        image_set_filenames=[VOC_2013_trainval_image_set_filename],
                        # sample_weights_dirs=VOC_2013_sampleweights_dir,
                        annotations_dirs=[VOC_2013_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)
val_dataset.parse_xml(images_dirs=[VOC_2013_images_dir],
                      image_set_filenames=[VOC_2013_test_image_set_filename],
                      annotations_dirs=[VOC_2013_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)


batch_size = 8
# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)
# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)
# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('deconv1_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales_coco,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)


train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()

val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:/t{:>6}".format(train_dataset_size))

print("Number of images in the validation dataset:/t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 120:
        return 0.0001
    else:
        return 0.00001

# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='D:/Proposed_model/6typedefectWeight/ssd512_6DEFECTdataset_epoch-{epoch:02d}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd512_6defects_training_log_7_21.csv',
                       separator=',',
                       append=True)
learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 30
steps_per_epoch = 100

history = model.fit_generator(generator=train_generator,
                              class_weight='auto',
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)
elapsed_time = time.time()
print("elapsed_time:{0}".format(elapsed_time)+"[sec]")

#plot Perfomance
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc','val_acc'], loc= 'lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['losss', 'val_loss'], loc = 'upper left')
plt.show()


model_name = 'ssd512_6defect_2fc7decon1conv789'
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))

print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()



predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)
# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))

# 3: Make predictions.


y_pred = model.predict(batch_images)

# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
# 5: Convert the predictions for the original image.

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])


# 5: Draw the predicted boxes onto the image

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()


classes = ['background','rolled-in_scale','scratches','pitted_surface','patches','inclusion','crazing']



plt.imshow(batch_images[i])
current_axis = plt.gca()

plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_pred_decoded[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()



