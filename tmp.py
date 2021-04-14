import numpy as np
import matplotlib.pyplot as plt


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))  # turns image into 28x28 matrix
    plt.imshow(image, 'gray')  # displays matrix as grayscale


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def sigmoid_deriv(array):
    return sigmoid(array) * (1 - sigmoid(array))


# 1st Step
# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256  # Initializes pixels

    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1  # Label is now 10x1 array which shows the desired output
    train_set.append((image, label))

# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256  # Initializes pixels

    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1  # Label is now 10x1 array which shows the desired output
    test_set.append((image, label))

# Plotting an image
show_image(train_set[0][0])
plt.ylabel(np.where(train_set[0][1] == [1.])[0][0])
plt.show()
show_image(train_set[4][0])
plt.ylabel(np.where(train_set[4][1] == [1.])[0][0])
plt.show()

W_L1 = np.random.normal(loc=0, scale=1, size=(16, 784))
B_L1 = np.zeros((16, 1))
W_L2 = np.random.normal(loc=0, scale=1, size=(16, 16))
B_L2 = np.zeros((16, 1))
W_L3 = np.random.normal(loc=0, scale=1, size=(10, 16))
B_L3 = np.zeros((10, 1))

batch_size = 50
learning_rate = 1
number_of_epochs = 5

costs = []

for n in range(number_of_epochs):
    acc = 0
    cost = 0

    np.random.shuffle(train_set)

    for i in range(int(60000 / batch_size)):
        grad_W_L1 = np.zeros((16, 784))
        grad_B_L1 = np.zeros((16, 1))
        grad_W_L2 = np.zeros((16, 16))
        grad_B_L2 = np.zeros((16, 1))
        grad_W_L3 = np.zeros((10, 16))
        grad_B_L3 = np.zeros((10, 1))
        grad_a_L2 = np.zeros((16, 1))
        grad_a_L1 = np.zeros((16, 1))
        for b in range(batch_size):
            a_L0 = np.asarray(train_set[i*batch_size + b][0])
            z_L1 = np.matmul(W_L1, a_L0) + B_L1
            a_L1 = sigmoid(z_L1)  # The second layer of neurons are initialized
            z_L2 = np.matmul(W_L2, a_L1) + B_L2
            a_L2 = sigmoid(z_L2)  # The third layer of neurons are initialized
            z_L3 = np.matmul(W_L3, a_L2) + B_L3
            a_L3 = sigmoid(z_L3)  # The last layer of neurons are initialized

            cost += sum((a_L3 - train_set[i*batch_size + b][1]) ** 2)

            # Grad hesab kardan!
            y = train_set[i*batch_size + b][1]

            grad_W_L3 += np.matmul((2 * (a_L3 - y) * sigmoid_deriv(z_L3)), np.transpose(a_L2))
            grad_B_L3 += (2 * (a_L3 - y) * sigmoid_deriv(z_L3))
            grad_a_L2 = np.matmul(np.transpose(W_L3), 2 * (a_L3 - y) * sigmoid_deriv(z_L3))
            grad_W_L2 += np.matmul((grad_a_L2 * sigmoid_deriv(z_L2)), np.transpose(a_L1))
            grad_B_L2 += (grad_a_L2 * sigmoid_deriv(z_L2))
            grad_a_L1 = np.matmul(np.transpose(W_L2), grad_a_L2 * sigmoid_deriv(z_L2))
            grad_W_L1 += np.matmul((grad_a_L1 * sigmoid_deriv(z_L1)), np.transpose(a_L0))
            grad_B_L1 += (grad_a_L1 * sigmoid_deriv(z_L1))


            x = 0
            num = 0
            # Find neuron with max activation between output neurons
            for k in range(10):
                if a_L3[k] > num:
                    num = a_L3[k]
                    x = k  # Save its index
            if train_set[i*batch_size + b][1][x] == 1:
                acc += 1  # If we predicted the same as label then increment accuracy by 1

        W_L1 = W_L1 - (learning_rate * (grad_W_L1 / batch_size))
        B_L1 = B_L1 - (learning_rate * (grad_B_L1 / batch_size))
        W_L2 = W_L2 - (learning_rate * (grad_W_L2 / batch_size))
        B_L2 = B_L2 - (learning_rate * (grad_B_L2 / batch_size))
        W_L3 = W_L3 - (learning_rate * (grad_W_L3 / batch_size))
        B_L3 = B_L3 - (learning_rate * (grad_B_L3 / batch_size))

    costs.append(cost/60000)
    print("Accuracy after epoch.", n+1, "is", acc/60000, "%")

plt.title("Final Step graph")
plt.xlabel("Epoch")
plt.ylabel("Cost Function")
x = np.arange(0, 5)
plt.plot(x, costs, color="blue")
plt.savefig("cost_plot.png")
