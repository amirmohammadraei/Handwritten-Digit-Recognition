import numpy as np
import matplotlib.pyplot as plt


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def sigmoid(seq):
    result = 1 / (1 + np.exp(-seq))
    return result


def multiply_mtx(w, mtx, b):
    result = w @ mtx + b
    return result


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


# Second step
w1 = np.random.normal(loc=0, scale=1, size=(16, 28*28))
w2 = np.random.normal(loc=0, scale=1, size=(16, 16))
w3 = np.random.normal(loc=0, scale=1, size=(10, 16))

b1 = np.zeros((16,1))
b2 = np.zeros((16,1))
b3 = np.zeros((10,1))

accuracy = 0

for i in range(0, 100):
    main_mtx = []
    for j in range(28*28):
        main_mtx.append(train_set[i][0][j]) 
    main_mtx = np.asarray(main_mtx)
    mtx2 = sigmoid(multiply_mtx(w1, main_mtx, b1))
    mtx3 = sigmoid(multiply_mtx(w2, mtx2, b2))
    f_mtx = sigmoid(multiply_mtx(w3, mtx3, b3))
    
    max_value = np.max(f_mtx)
    index_max_value = np.argmax(f_mtx)

    if train_set[i][1][index_max_value] == 1:
        accuracy += 1


print(f"Accuracy is: {accuracy}%")