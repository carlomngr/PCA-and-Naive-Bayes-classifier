from PIL import Image
import numpy as np
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_image (infilename):

    img = Image.open(infilename)
    data = np.asarray(img, dtype=np.float64)
    data = data.ravel()
    return data

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
X_std = (X - np.mean(X, axis=0))/(np.std(X, axis=0))

#Compute the PCA I need
pca = PCA()
X_t = pca.fit(X_std)
pca60 = X_t.components_[0:60, :]
pca6 = pca60[0:6, :]
pca2 = pca6[0:2, :]
last6 = X_t.components_[-6:, :]

#Trasform the data
trasformed2 = my_trasform(X_std, X_t, pca2)
trasformed6 = my_trasform(X_std, X_t, pca6)
trasformed60 = my_trasform(X_std, X_t, pca60)
trasformedlast6 = my_trasform(X_std, X_t, last6)

#Inverse trasform the data
trasformed60 = my_inverse_trasform(trasformed60, X_t, pca60)
trasformed6 = my_inverse_trasform(trasformed6, X_t, pca6)
trasformed2 = my_inverse_trasform(trasformed2, X_t, pca2)
trasformedlast6 = my_inverse_trasform(trasformedlast6, X_t, last6)

#Destandardizise matrix
trasformed60 = trasformed60 * np.std(X, axis=0) + np.mean(X, axis=0)
trasformed6 = trasformed6 * np.std(X, axis=0) + np.mean(X, axis=0)
trasformed2 = trasformed2 * np.std(X, axis=0) + np.mean(X, axis=0)
trasformedlast6 = trasformedlast6 * np.std(X, axis=0) + np.mean(X, axis=0)

#Plotting reduced images
img60 = np.reshape(trasformed60[0], (227, 227, 3)).astype(int)
img6 = np.reshape(trasformed6[0], (227, 227, 3)).astype(int)
img2 = np.reshape(trasformed2[0], (227, 227, 3)).astype(int)
imglast6 = np.reshape(trasformedlast6[0], (227, 227, 3)).astype(int)
img = np.reshape(X[0], (227, 227, 3)).astype(int)

fig = plt.figure()
columns = 2
rows = 3

fig.add_subplot(rows, columns, 1).set_title('Real image')
plt.imshow(img)
fig.add_subplot(rows, columns, 2).set_title('PCA 60')
plt.imshow(img60)
fig.add_subplot(rows, columns, 3).set_title('PCA 6')
plt.imshow(img6)
fig.add_subplot(rows, columns, 4).set_title('PCA 2')
plt.imshow(img2)
fig.add_subplot(rows, columns, 5).set_title('Last 6 PCA')
plt.imshow(imglast6)
plt.show()
plt.close()






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