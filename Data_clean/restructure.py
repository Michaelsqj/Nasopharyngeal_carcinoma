import json
import sys
import os
import shutil
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger('test')
logger.setLevel(level=logging.DEBUG)
logging.basicConfig(
    format=
    '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.DEBUG,
    filename='test.log',
    filemode='w')
formatter = logging.Formatter('%(levelname)s: %(message)s')

file_handler = logging.FileHandler('annot_image.log', mode='w')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def extract_annot(name_mapping=None):
    with open('./annotation.json', 'r') as f:
        datalist = json.load(f)
    os.chdir(r'E:\NoseCancer\鼻咽部病例-建模型用的副本')
    ####
    ##datalist[i] : dict_keys(['json_format_version', 'task', 'patientID', 'studyUID', 'seriesUID', 'quality', 'slice_spacing', 'slice_thickness', 'pixel_spacing', 'other_info', 'nodes'])
    ####
    ##patienID=./非鼻咽癌/建模型用非鼻咽癌病例/正常鼻咽部病例/正常鼻咽-傅劲松
    ####
    ##nodes[i] : dict_keys(['type', 'note', 'node_index', 'confidence', 'from', 'attr', 'rois', 'bounds', 'descText', 'mark_num'])
    ####
    ##node['rois'] : (1)[dict,dict,dict] -->> dict_keys(['slice_index', 'edge'])  -->>image_name , annotation  (2)[] 空 node['bound']非空
    ####
    ##annot['nodes'][0]['rois'][i]['edge']
    ####
    data_dir = r'E:\NoseCancer\Dataset'
    count = 0
    for annot in datalist:

        temp = str.split(annot['patientID'], '/')
        if temp[-1] == 'Images':
            patient = str.split(temp[-2], '/')[-1]
        else:
            patient = temp[-1]

        # if len(annot['nodes']) > 0:
        print(annot['patientID'], len(annot['nodes']))

        i = 0
        for node in annot['nodes']:
            # print(i)
            i += 1
            # if str(type(node['rois'])) == '<class \'list\'>':
            if len(node['rois']) > 0:
                # print(annot['patientID'], [node['rois'][i]['slice_index'] for i in range(len(node['rois']))])
                print([
                    len(node['rois'][j]['edge'])
                    for j in range(len(node['rois']))
                ])
                pass
            else:
                # print(node['bounds'][0]['slice_index'])
                # pass
                pass
                # print(len(node['rois']))
                # print('count:', count)
                # count += 1


def rename_move_img():
    ##对每个病例小文件夹重命名，建立映射dict npy文件   命名：病人_图片
    ##{new_name: [patient_name,image_name,imgpath,image_shape]}
    data_dir = r'E:\NoseCancer\Dataset'
    map_file = r'E:\NoseCancer\Dataset\name_mapping.npy'
    map_dict = {}
    dir1 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\鼻咽癌\建模鼻咽癌病例-无录像'
    dir2 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\鼻咽癌\建模鼻咽癌病例-有录像'
    dir3 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\建模型用鼻咽部炎症病例-无录像'
    dir4 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\建模型用鼻咽部炎症病例-有录像'
    dir5 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\建模型用-鼻咽淋巴瘤'
    dir6 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\正常鼻咽部病例'
    ##image=cv2.imdecode(np.fromfile(imgpath,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    ##图片[(576, 768, 4), (1080, 1920, 4), (562, 752, 4)]  诊断(1753, 1240, 4)
    Dir = [dir1, dir2, dir3, dir4, dir5, dir6]
    image_shape = []
    start_time = time.time()
    Count_patient = 0
    for d in Dir:
        for patient in os.listdir(d):
            Count_image = 0
            Count_video = 0
            for file in os.listdir(os.path.join(d, patient)):
                ##(1)directory (Images,Videos)
                ##(2)*.bmp
                ##(3)*.wmv
                if file == 'Images':
                    for img in os.listdir(os.path.join(d, patient, file)):
                        imgpath = os.path.join(d, patient, file, img)
                        image = cv2.imdecode(
                            np.fromfile(imgpath, dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED)
                        if image.shape not in image_shape:
                            image_shape.append(image.shape)
                            print(image.shape, imgpath)
                        map_dict[f'{Count_patient}_{Count_image}'] = [
                            imgpath, image.shape
                        ]
                elif file[-3:] == 'bmp':
                    imgpath = os.path.join(d, patient, file)
                    image = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8),
                                         cv2.IMREAD_UNCHANGED)
                    if image.shape not in image_shape:
                        image_shape.append(image.shape)
                        print(image.shape, imgpath)
                    map_dict[f'{Count_patient}_{Count_image}'] = [
                        imgpath, image.shape
                    ]
                elif file[-3:] == 'wmv':
                    # shutil.copy(os.path.join(d, patient, file), os.path.join(data_dir,'Videos',f'{Count_patient}_{Count_video}.{file[-3:]}'))
                    Count_video += 1
                elif file == 'Videos':
                    ## mp4,wmv,txt
                    for video in os.listdir(os.path.join(d, patient, file)):
                        if video[-3:] == 'wmv' or video[-3:] == 'mp4':
                            # shutil.copy(os.path.join(d, patient, file, video), os.path.join(data_dir,'Videos',f'{Count_patient}_{Count_video}.{video[-3:]}'))
                            Count_video += 1
                else:
                    print('Not image or video', d, patient, file)
                    pass
            Count_patient += 1
    print(image_shape)
    print(time.time() - start_time)
    print(len(map_dict))
    np.save(map_file, map_dict)


def comp_img_annot():
    ####找出没有标记的图片
    with open('./annotation.json', 'r') as f:
        datalist = json.load(f)
    annot_patients = []
    for annot in datalist:
        temp = str.split(annot['patientID'], '/')
        if temp[-1] == 'Images':
            patient = temp[-2]
        else:
            patient = temp[-1]
        if patient not in annot_patients:
            annot_patients.append(patient)
            print(patient)
    # print(len(annot_patients))
    dir1 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\鼻咽癌\建模鼻咽癌病例-无录像'
    dir2 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\鼻咽癌\建模鼻咽癌病例-有录像'
    dir3 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\建模型用鼻咽部炎症病例-无录像'
    dir4 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\建模型用鼻咽部炎症病例-有录像'
    dir5 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\建模型用-鼻咽淋巴瘤'
    dir6 = r'E:\NoseCancer\鼻咽部病例-建模型用的副本\非鼻咽癌\建模型用非鼻咽癌病例\正常鼻咽部病例'
    Dir = [dir1, dir2, dir3, dir4, dir5, dir6]
    patients = []
    for d in Dir:
        for patient in os.listdir(d):
            if patient not in patients:
                patients.append(patient)
    print(len(patients))
    for p in annot_patients:
        if p not in patients:
            print(p)

    ##结果 标注171 病例170 侯春龙_01.201807230067.01有标注，无图片


def classify(path):
    '''

    :param path: e.g ./非鼻咽癌/建模型用非鼻咽癌病例/正常鼻咽部病例/正常鼻咽-傅劲松
    :return:
    '''
    C0 = '鼻咽癌'
    C1 = '炎症'
    C2 = '淋巴瘤'
    C3 = '正常'
    if C1 in path:
        return 1
    elif C2 in path:
        return 2
    elif C3 in path:
        return 3
    elif C0 in path:
        return 0
    else:
        raise ValueError('Path wrong')


def annot_image():
    with open('./annotation.json', 'r') as f:
        datalist = json.load(f)
    os.chdir(r'E:\NoseCancer\鼻咽部病例-建模型用的副本')
    ##图片[(576, 768, 4), (1080, 1920, 4), (562, 752, 4)]  诊断(1753, 1240, 4)
    image_shape = [(576, 768, 4), (1080, 1920, 4), (562, 752, 4),
                   (1753, 1240, 4)]
    ####
    ##datalist[i] : dict_keys(['json_format_version', 'task', 'patientID', 'studyUID', 'seriesUID', 'quality', 'slice_spacing', 'slice_thickness', 'pixel_spacing', 'other_info', 'nodes'])
    ####
    ##patienID=./非鼻咽癌/建模型用非鼻咽癌病例/正常鼻咽部病例/正常鼻咽-傅劲松
    ####
    ##nodes[i] : dict_keys(['type', 'note', 'node_index', 'confidence', 'from', 'attr', 'rois', 'bounds', 'descText', 'mark_num'])
    ####
    ##node['rois'] : (1)[dict,dict,dict] -->> dict_keys(['slice_index', 'edge'])  -->>image_name , annotation  (2)[] 空 node['bound']非空
    ####
    ##annot['nodes'][0]['rois'][i]['edge']
    ####
    data_dir = r'E:\NoseCancer\Dataset'
    map_dict = {}
    patient_images = {}  ## patient:{ImageName:[edge,edge]}
    Count_patient = 0
    for annot in datalist:
        start_time = time.time()
        temp = str.split(annot['patientID'], '/')
        if temp[-1] == 'Images':
            patient = temp[-2]
            path = annot['patientID'][:-7]
        else:
            patient = temp[-1]
            path = annot['patientID']

        ####小文件夹下所有的annotation
        patient_images[patient] = {}  ## ImageName:[rois,rois]
        for node in annot['nodes']:
            if len(node['rois']) > 0:
                for roi in node['rois']:
                    if roi['slice_index'] not in patient_images[patient].keys(
                    ):
                        patient_images[patient][roi['slice_index']] = []
                        patient_images[patient][roi['slice_index']].append(
                            roi['edge'])
                    else:
                        patient_images[patient][roi['slice_index']].append(
                            roi['edge'])
            else:
                pass
                # print(len(node['rois']))
                # print('count:', count)
                # count += 1

        ####进入小文件夹，所有的img和video
        Count_image = 0
        Count_video = 0
        No_Annot = []
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            ##(1)directory (Images,Videos)
            ##(2)*.bmp
            ##(3)*.wmv
            '''
            Annotation patient_image.npy 文件 {'class': ,'edge':[]}
            '''
            if file == 'Images':
                for img in os.listdir(os.path.join(path, file)):
                    imgpath = os.path.join(path, file, img)
                    image = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8),
                                         cv2.IMREAD_UNCHANGED)
                    if image.shape != (1753, 1240, 4):
                        C = classify(path)
                        annotation = {}
                        annotation['class'] = C
                        if img in patient_images[patient].keys():
                            annotation['edge'] = patient_images[patient][img]
                        else:
                            annotation['edge'] = []
                            No_Annot.append(imgpath)
                        map_dict[f'{Count_patient}_{Count_image}'] = [
                            imgpath, image.shape
                        ]
                        shutil.copy(
                            imgpath,
                            os.path.join(data_dir, 'Images',
                                         f'{Count_patient}_{Count_image}.bmp'))
                        np.save(
                            os.path.join(data_dir, 'Annotations',
                                         f'{Count_patient}_{Count_image}.npy'),
                            annotation)
                        Count_image += 1
                    elif image.shape == (1753, 1240, 4):
                        ##诊断文件
                        shutil.copy(
                            imgpath,
                            os.path.join(data_dir, 'Reports',
                                         f'{Count_patient}.bmp'))
            elif file[-3:] == 'bmp':
                imgpath = os.path.join(path, file)
                image = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8),
                                     cv2.IMREAD_UNCHANGED)
                if image.shape != (1753, 1240, 4):
                    C = classify(path)
                    annotation = {}
                    annotation['class'] = C
                    if file in patient_images[patient].keys():
                        annotation['edge'] = patient_images[patient][file]
                    else:
                        annotation['edge'] = []
                        No_Annot.append(imgpath)
                    map_dict[f'{Count_patient}_{Count_image}'] = [
                        imgpath, image.shape
                    ]
                    shutil.copy(
                        imgpath,
                        os.path.join(data_dir, 'Images',
                                     f'{Count_patient}_{Count_image}.bmp'))
                    np.save(
                        os.path.join(data_dir, 'Annotations',
                                     f'{Count_patient}_{Count_image}.npy'),
                        annotation)
                    Count_image += 1
                elif image.shape == (1753, 1240, 4):
                    ##诊断文件
                    shutil.copy(
                        imgpath,
                        os.path.join(data_dir, 'Reports',
                                     f'{Count_patient}.bmp'))
            elif file[-3:] == 'wmv':
                shutil.copy(
                    os.path.join(path, file),
                    os.path.join(data_dir, 'Videos',
                                 f'{Count_patient}_{Count_video}.{file[-3:]}'))
                Count_video += 1
            elif file == 'Videos':
                ## mp4,wmv,txt
                for video in os.listdir(os.path.join(path, file)):
                    if video[-3:] == 'wmv' or video[-3:] == 'mp4':
                        shutil.copy(
                            os.path.join(path, file, video),
                            os.path.join(
                                data_dir, 'Videos',
                                f'{Count_patient}_{Count_video}.{video[-3:]}'))
                        Count_video += 1
            else:
                logger.warning('Not image or video  ' + str(path) + '  ' +
                               str(file))
                pass

        logger.info(
            str(Count_patient) + '  ' + str(patient) + '   %.2f s' %
            (time.time() - start_time) + '  No Annot:  ' + str(No_Annot))
        Count_patient += 1
    np.save(r'E:\NoseCancer\Dataset\name_mapping.npy', map_dict)


def split_nbi():
    '''
    0_0 (576, 768, 3)   (:73,710:)
    3_0 (562, 752, 4)   (:88,712:)
    4_0 (576, 768, 4)   (:69,712:)
    12_0 (1080, 1920, 4) (1600-1688,50)
    '''
    data_dir = r'E:\NoseCancer\Dataset'
    os.chdir(data_dir)
    image_shape = []
    p1 = cv2.imread(os.path.join('Images', '0_2.bmp'))[:73, 710:, :]
    p2 = cv2.imread(os.path.join('Images', '3_6.bmp'))[:88, 712:, :]
    p3 = cv2.imread(os.path.join('Images', '4_6.bmp'))[:69, 712:, :]
    p4 = cv2.imread(os.path.join('Images', '12_7.bmp'))[1600:1688, :50, :]
    # name_mapping = np.load('name_mapping.npy', allow_pickle=True).item()
    # for name in name_mapping.keys():
    #     shape = name_mapping[name][1]
    #     if shape not in image_shape:
    #         image_shape.append(shape)
    #         print(name, name_mapping[name][1])
    for img in os.listdir('Images'):
        image = cv2.imread(os.path.join('Images', img))
        # annot = np.load(os.path.join('Annotations', img[:-3] + 'npy'),allow_pickle=True).item()
        if image.shape == (576, 768, 3):
            if (image[:73, 710:, :] == p1).all():
                C = 'nbi'
            else:
                C = 'white'
        elif image.shape == (562, 752, 4):
            if (image[:88, 712:, :] == p2).all():
                C = 'nbi'
            else:
                C = 'white'
        elif image.shape == (576, 768, 4):
            if (image[:69, 712:, :] == p3).all():
                if (image[:88, 712:, :] == p2).all():
                    C = 'nbi'
                else:
                    C = 'white'
        elif image.shape == (1080, 1920, 4):
            if (image[1600:1688, :50, :] == p4).all():
                if (image[:88, 712:, :] == p2).all():
                    C = 'nbi'
                else:
                    C = 'white'
        shutil.copyfile(os.path.join(data_dir, 'Images', img),
                        os.path.join(data_dir, C, img))
        print(img)


def comp():
    data_dir = r'E:\NoseCancer\Dataset'
    os.chdir(data_dir)
    p1 = cv2.imread(os.path.join('Images', '0_2.bmp'))[:73, 710:, :]
    p2 = cv2.imread(os.path.join('Images', '1_2.bmp'))[:73, 710:, :]
    cv2.imshow('1', p1)
    cv2.imshow('2', p2)
    p = p1 - p2
    cv2.waitKey(0)


# comp_img_annot()
# rename_move_img()
# extract_annot()
# annot_image()
# split_nbi()
comp()