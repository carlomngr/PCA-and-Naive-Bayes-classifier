from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

def load_image (infilename):

    img = Image.open(infilename)
    data = np.asarray(img, dtype=np.float64)
    data = data.ravel()
    return data

folders = ['dog' , 'guitar' , 'house' , 'person']
X = []
for folder in folders:
    for infile in glob.glob(folder + "\*.jpg"):
        img = load_image(infile)
        X.append(img)
X = np.array(X)