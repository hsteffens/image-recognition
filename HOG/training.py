import cv2
import pandas as pd
import numpy
import os


def getdataset(items, path, size, classification):
    for x in range(1, size):
        image = cv2.imread(path + '{:03d}'.format(x) + '.bmp')
        if image is None:
            continue
        dim = (64, 128)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        hog = cv2.HOGDescriptor()
        vector = hog.compute(image)
        vector = numpy.append(vector, [classification])
        items.append(vector)
    return

def createDataFile(dataset, filename):
    outdir = './dir'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, filename)

    dataset.to_csv(fullname)


def createdata():
    training = list()
    getdataset(training, '../Treinamento/bart', 81, 0)
    getdataset(training, '../Treinamento/homer', 63, 1)

    dataset = pd.DataFrame.from_records(training)
    createDataFile(dataset, 'hog.csv')

    dataTest = list()
    getdataset(dataTest, '../Teste/bart', 116, 0)
    getdataset(dataTest, '../Teste/homer', 88, 1)

    dataset = pd.DataFrame.from_records(dataTest)
    createDataFile(dataset, 'test_hog.csv')