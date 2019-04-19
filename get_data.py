import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt

def cvtRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

def get_image(path ,img_width = 48, img_height = 48):
    img = cv2.imread(path)

    plt.imshow(cvtRGB(img))

    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,48,48,3).astype('float32')

    return img

def find_category(key = -1):
    categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    mapping = {-1:"not available. Sorry!"}
    count = 0
    for i in categories:
        mapping[count] = i
        count+=1
    return mapping[key]
def get_dataset(train_path = 'flower_photos/train_data/', base_path = 'flower_photos/train_data/'):

    categories = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    count = 0

    """### Load File Names"""

    fnames = []
    for category in categories:
        flower_folder = os.path.join(base_path, category)
        file_names = os.listdir(flower_folder)
        full_path = [os.path.join(flower_folder, file_name) for file_name in file_names]
        fnames.append(full_path)

    """### Load Images"""

    images = []
    for names in fnames:
        one_category_images = [cv2.imread(name) for name in names if (cv2.imread(name)) is not None]
        images.append(one_category_images)

    print('number of images for each category:', [len(f) for f in images])

    """### Finding Minimum shape for each category"""

    for i,imgs in enumerate(images):
        shapes = [img.shape for img in imgs]
        widths = [shape[0] for shape in shapes]
        heights = [shape[1] for shape in shapes]
        print('%d,%d is the min shape for %s' % (np.min(widths), np.min(heights), categories[i]))

    """### Resizing Images"""

    img_width, img_height = 48, 48

    resized_images = []
    for i,imgs in enumerate(images):
        resized_images.append([cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC) for img in imgs])

    """### Splitting into training and testing set"""

    from sklearn.model_selection import train_test_split
    train_images = []
    val_images = []
    for imgs in resized_images:
        train, test = train_test_split(imgs, train_size=0.9, test_size=0.1)
        train_images.append(train)
        val_images.append(test)

    """### Create Labels"""

    train_labels = np.zeros(sum([len(i) for i in train_images]))
    count = 0
    next_class = 0
    for i in train_images:
        train_labels[next_class:next_class+len(i)] = count
        next_class += len(i)
        count+=1
        
    val_labels = np.zeros(sum([len(i) for i in val_images]))
    count = 0
    next_class = 0
    for i in val_images:
        val_labels[next_class:next_class+len(i)] = count
        next_class += len(i)
        count+=1

    """### Converting Image data to numpy array"""
    tmp_train_imgs = []
    tmp_val_imgs = []
    for imgs in train_images:
        tmp_train_imgs += imgs
    for imgs in val_images:
        tmp_val_imgs += imgs
    train_images = np.array(tmp_train_imgs)
    val_images = np.array(tmp_val_imgs)

    train_data = train_images.astype('float32')
    val_data = val_images.astype('float32')
    train_labels = keras.utils.to_categorical(train_labels, 5)
    val_labels = keras.utils.to_categorical(val_labels, 5)

    return train_data,val_data,train_labels,val_labels