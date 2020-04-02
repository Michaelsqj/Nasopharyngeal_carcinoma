import numpy as np
import os
import cv2
import logging
logger = logging.getLogger('')
image_dir = r'E:\NoseCancer\Dataset\images\nbi'
annot_dir = r'E:\NoseCancer\Dataset\Annotations'
imgout = r'E:\NoseCancer\Dataset\nbi'
anout = r'E:\NoseCancer\Dataset\an'
for file in os.listdir(image_dir):
    name = str.split(file, '.')[0]
    image = cv2.imread(os.path.join(image_dir, file))
    annot = np.load(os.path.join(annot_dir, name + '.npy'),
                    allow_pickle=True).item()
    ##图片[(576, 768, 4), (1080, 1920, 4), (562, 752, 4)]
    # (60:540,274:762) (13:941,666:1594) (78:558,273:)
    height, width, _ = image.shape
    edge = annot['edge']
    new_edge = []
    if (height, width) == (576, 768):
        image = image[60:541, 274:763, :]
        if len(edge) > 0:
            for e in edge:
                e = np.array(e) - np.array([274, 60])
                # cv2.polylines(image, [e], True, 255)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                new_edge.append(e.tolist())

    elif (height, width) == (1080, 1920):
        image = image[13:942, 666:1595, :]
        if len(edge) > 0:
            for e in edge:
                e = np.array(e) - np.array([666, 13])
                # cv2.polylines(image, [e], True, 255)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                new_edge.append(e.tolist())
    elif (height, width) == (562, 752):
        image = image[78:559, 273:, :]
        if len(edge) > 0:
            for e in edge:
                e = np.array(e) - np.array([273, 78])
                # cv2.polylines(image, [e], True, 255)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                new_edge.append(e.tolist())
    else:
        print(file, len(edge))
        print('edge wrong')
    annot['edge'] = new_edge
    cv2.imwrite(os.path.join(imgout, file), image)
    np.save(os.path.join(anout, name + '.npy'), annot)
