import cv2
import pandas as pd
import numpy
import os


def getdataset(items, path, size):
    for x in range(1, size):
        image = cv2.imread(path + '{:03d}'.format(x) + '.bmp')

        if image is None:
            continue
        dim = (76, 128)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(image, None)
        # compute the descriptors with ORB
        kp, descriptors = orb.compute(image, kp)
        if 65 < descriptors.shape[0]:
            descriptors = numpy.delete(descriptors, 65, 0)
        items.append(descriptors.reshape(descriptors.shape[0] * descriptors.shape[1]))
    return

def createDataFile(dataset, filename):
    outdir = './dir'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, filename)

    dataset.to_csv(fullname)


def createdata():
    training = list()
    getdataset(training, '../Treinamento/bart', 81)
    getdataset(training, '../Treinamento/homer', 63)

    dataset = pd.DataFrame.from_records(training)
    createDataFile(dataset, 'orb.csv')

    dataTest = list()
    getdataset(dataTest, '../Teste/bart', 116)
    getdataset(dataTest, '../Teste/homer', 88)

    dataset = pd.DataFrame.from_records(dataTest)
    createDataFile(dataset, 'test_orb.csv')
