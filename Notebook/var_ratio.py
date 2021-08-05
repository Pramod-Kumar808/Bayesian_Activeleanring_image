import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

# set some variables
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)

all_test_acc = []
all_test_loss = []

batch_samples = 1000
amount_used_samples = 200
start_amount = 200

def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 3,000 images to evaluate how accurately the network learned to classify images.
    """

    datasets = ['seg_train', 'seg_test']
    output = []

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()
# shuffle the images so they are not in the same orders
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

# Perma dropout in order to make dropout available during inference runs
from keras.layers.core import Lambda
from keras import backend as K
# Make model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(PermaDropout(0.5))
model.add(MaxPooling2D(2,2))
model.add(PermaDropout(0.5))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(PermaDropout(0.5))
model.add(MaxPooling2D(2,2))
model.add(PermaDropout(0.5))
model.add(Flatten())
model.add(PermaDropout(0.5))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(6, activation=tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Get a random sample of images to train on
import random
labelled_indices = random.sample(range(train_images.shape[0]), start_amount)

# Get the images that will be used as labelled images
train_images_labelled = train_images[labelled_indices, :, :, :]

# Get the appropriate labels
train_labels_labelled = train_labels[labelled_indices]

# Get the pool images (this is the pool with all the unlabelled images)
pool_images = np.delete(train_images, labelled_indices, axis=0)

# Get the labels from all the images in the pool
pool_labels = np.delete(train_labels, labelled_indices, axis=0)

history = model.fit(train_images_labelled, train_labels_labelled, batch_size=128, epochs=20, validation_split = 0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

all_test_acc.append(test_acc)
all_test_loss.append(test_loss)


def mode(array):
    y = np.bincount(array)
    return np.argmax(y), y[np.argmax(y)]


for i in range(5):
    print('Loop: ', i)
    # Get a random batch out of the pool. from this batch, a few will be selected to add to the labelled images
    batch_indices = random.sample(range(pool_images.shape[0]), batch_samples)

    pool_batch_images = pool_images[batch_indices, :, :, :]
    pool_batch_labels = pool_labels[batch_indices]
    pool_images = np.delete(pool_images, batch_indices, axis=0)
    pool_labels = np.delete(pool_labels, batch_indices, axis=0)

    all_predictions = np.zeros(shape=(pool_batch_images.shape[0], 1))
    for i in range(10):
        predictions = model.predict(pool_batch_images)
    #     print(predictions)
        predictions = predictions.argmax(axis=1)
        predictions = predictions[..., np.newaxis]
    #     print(predictions)
    #     all_predictions[:, :-1] = predictions
        all_predictions = np.append(all_predictions, predictions, axis=1)

    variation = np.zeros(shape=(pool_batch_images.shape[0]))

    for t in range(pool_batch_images.shape[0]):  # for every row in all_predictions (so every data point in pool subset)
        L = np.array([0])
        for i in range(10):
            L = np.append(L, all_predictions[t, i + 1])

        Predicted_Class, Mode = mode(L[1:].astype(int))
        # print(Predicted_Class, Mode)
        v = np.array([1 - Mode / float(10)])
        #     print(v)
        variation[t] = v

    variation_fl = variation.flatten()
    pool_index_toadd = variation_fl.argsort()[::-1][:amount_used_samples]

    images_toadd = pool_batch_images[pool_index_toadd, :, :, :]

    labels_toadd = pool_batch_labels[pool_index_toadd]

    # Delete the images that are added to the labelled group
    pool_batch_images = np.delete(pool_batch_images, pool_index_toadd, axis=0)

    # Delete the labels that are added to the labelled group
    pool_batch_labels = np.delete(pool_batch_labels, pool_index_toadd, axis=0)

    # Add the images from the batch to the pool with unlabelled images
    pool_images = np.append(pool_images, pool_batch_images, axis=0)

    pool_labels = np.append(pool_labels, pool_batch_labels, axis=0)

    train_images_labelled = np.append(train_images_labelled, images_toadd, axis=0)

    train_labels_labelled = np.append(train_labels_labelled, labels_toadd, axis=0)

    history = model.fit(train_images_labelled, train_labels_labelled, batch_size=128, epochs=20, validation_split = 0.2)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    all_test_acc.append(test_acc)
    all_test_loss.append(test_loss)

    print(all_test_acc)
    print(all_test_loss)

loop = [i for i in range(len(all_test_acc))]
plt.plot(loop, all_test_acc, label='test accuracy')
plt.plot(loop, all_test_loss, label='test loss')






