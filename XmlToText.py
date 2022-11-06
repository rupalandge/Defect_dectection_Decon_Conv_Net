import os
import random

if not os.path.exists('datasets2/VOCdevkit/VOC2007/ImageSets/Main/'):
    os.makedirs('datasets2/VOCdevkit/VOC2007/ImageSets/Main/')

xmlfilepath = r'G:/SWIPENet_master2/datasets2/VOCdevkit/VOC2007/Annotations'
saveBasePath = r"G:/SWIPENet_master2/datasets2/VOCdevkit/VOC2007/ImageSets/Main"

trainval_percent =0.7
train_percent =0.7
total_xml =os.listdir(xmlfilepath)
num=len(total_xml)




list =range(num)

tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval = random.sample(list,tv)
train =random.sample(trainval,tr)

print("train and val size",tv)
print("train size",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'),'w')
ftest = open(os.path.join(saveBasePath,'test.txt'),'w')
ftrain =open(os.path.join(saveBasePath,'train.txt'),'w')
fval=open(os.path.join(saveBasePath,'val.txt'),'w')


#
# for i in list:
#     name =total_xml[i][:-4]+'\n'
#     if i in trainval:
#         ftrainval.write('G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'+name)
#         if i in train:
#             ftrain.write('G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'+name)
#         else:
#             fval.write('G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'+name)
#     else:
#         ftest.write('G:/ssd_keras_1_master/datasets1/VOCdevkit/VOC2007/Annotations/'+name)


for i in list:
    name =total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
