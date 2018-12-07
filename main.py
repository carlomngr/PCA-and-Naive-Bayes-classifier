from PIL import Image
import numpy as np
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import linalg as la

def load_image( infilename ) :

    img = Image.open(infilename)
    data = np.asarray(img, dtype=np.float64)
    data = data.ravel()
    return data

def myPCA_fit(data, dims_rescaled_data):

    #m, n = data.shape
    mean = data.mean(axis=0)
    data -= mean
    L = np.dot(data, data.T)
    evals, evecs = la.eigh(L)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :dims_rescaled_data]
    components = np.dot(data.T, evecs)

    return components.T

def my_trasform(data, X_t, components):

    return np.dot(data-X_t.mean_, components.T)

def my_inverse_trasform(data_reduced, X_t, components):

    return np.dot(data_reduced, components) + X_t.mean_

#Acquiring images in X matrix
folders = ['dog' , 'guitar' , 'house' , 'person']
X = []
for folder in folders:
    for infile in glob.glob(folder + "\*.jpg"):
        img = load_image(infile)
        X.append(img)
X = np.array(X)

#Standardization of the matrix
X_std = (X - np.mean(X, axis=0)/np.std(X, axis=0))

#Compute the PCA I needd
pca = PCA()
X_t = pca.fit(X_std)
pca60 = X_t.components_[0:60, :]
pca6 = pca60[0:6, :]
pca2 = pca6[0:2, :]


#Trasform the data
trasformed2 = my_trasform(X_std, X_t, pca2)
trasformed6 = my_trasform(X_std, X_t, pca6)
trasformed60 = my_trasform(X_std, X_t, pca60)
trasformed602 = X_t.PCA(2)
#Inverse trasform the data
trasformed60 = my_inverse_trasform(trasformed60, X_t, pca60)
#trasformed602 =
#Destandardizise matrix
trasformed60 = trasformed60 * np.std(X, axis=0) + np.mean(X, axis=0)
#aaaaaa
#Plotting reduced images
img60 = np.reshape(trasformed60[0], (227, 227, 3)).astype(int)
plt.imshow(img60)
plt.show()
plt.close()
#matrix, autoval, autovett = myPCA(X_stand, 6)
#img2 =np.reshape(X[113], (227, 227, 3)).astype(int)
#plt.imshow(img2, interpolation='nearest')
#plt.show()
#X_t = pca2.fit(X_stand)

#X_t = my_trasform(X_stand, comp)
#X_reproj2 = my_inverse_trasform(X_t, comp)
#X_new2 = (X_reproj2 * np.std(X, axis=0) + np.mean(X, axis=0))
#img1 =np.reshape(X_new2[0], (227, 227, 3)).astype(int)
#plt.imshow(img1, interpolation='nearest')
#plt.show()
#plt.close()

#X_reproj = pca2.inverse_transform(X_t)
#X_new = (X_reproj * np.std(X, axis=0) + np.mean(X, axis=0))
#img1 =np.reshape(X_new, (227, 227, 3)).astype(int)
#img = numpy2pil(img1)

#plt.imshow(img1, interpolation='nearest')
#plt.show()
#plt.close()



dog = len(glob.glob('dog/*.jpg'))
guitar = len(glob.glob('guitar/*.jpg'))
house = len(glob.glob('house/*.jpg'))
person = len(glob.glob('person/*.jpg'))

y = ['r', 'c', 'b', 'g']
'''
single = X_t[0]

plt.title("Single image")
plt.scatter(single[0], single[1], c='r')
plt.show()
plt.close()

plt.title("Whole data-set")
plt.scatter(X_t[0:dog, 0], X_t[0:dog, 1], c=y[0])
plt.scatter(X_t[dog:dog+guitar, 0], X_t[dog:dog+guitar, 1], c=y[1])
plt.scatter(X_t[dog+guitar:dog+guitar+house, 0], X_t[dog+guitar:dog+guitar+house, 1], c=y[2])
plt.scatter(X_t[dog+guitar+house:dog+guitar+house+person, 0], X_t[dog+guitar+house:dog+guitar+house+person, 1], c=y[3])
plt.show()
plt.close()
print("ciao")'''