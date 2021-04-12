import numpy as np
import matplotlib.pyplot as plt

# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
train_images_file = open('samples/train-images-idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('samples/train-labels-idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1
    
    train_set.append((image, label))


# Reading The Test Set
test_images_file = open('samples/t10k-images-idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('samples/t10k-labels-idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1
    
    test_set.append((image, label))


# Plotting an image
#show_image(train_set[85][0])
#plt.show()





# first step
Bl0=np.zeros((16,1))
# b0
Al0= np.random.normal(loc=0, scale=1, size=(16, 28*28))
Al0
acc=0
a=[]
for i in range (100):
    a=[]
    for j in range(28*28):
        a.append(train_set[i][0][j]) 
    Al0= np.random.normal(loc=0, scale=1, size=(16, 28*28))
    a0=np.asarray(a) 
#     print(a0[550])
    a1=Al0@a0+Bl0
    #print(a1.shape)
    a1= 1/(1 + np.exp(-a1))
#     print(a1)
    Al1=np.random.normal(loc=0, scale=1, size=(16, 16))
    Bl1=np.zeros((16,1))
    a2=Al1@a1+Bl1
    a2= 1/(1 + np.exp(-a2))
#     print(a2)
    Al2=np.random.normal(loc=0, scale=1, size=(10, 16))
    Bl2=np.zeros((10,1))
    a3=Al2@a2+Bl2
    z = 1/(1 + np.exp(-a3))
#     print(z)
    x=0
    num=0
    for k in range (10):
        if z[k]>num:
            num=z[k]
            x=k
    if train_set[i][1][x]==1:
#         print(z[x])
        print(i)
        acc+=1
      
                
acc/100