from get_data import get_dataset
from get_model import get_model, save_model
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



def plot(img,labels,figsize = (14,8),rows = 1, interp = False, titles =None):
    if type(img[0]) is np.ndarray:
        img = np.array(img).astype(np.uint8)
        if img.shape[-1] != 3:
            img = img.transpose((0,2,3,1))
    f = plt.figure(figsize=(26,18))
    cols = len(img)//rows if len(img) % 2==0 else len(img)//rows +1
    for i in range(len(img)):
        sp = f.add_subplot(rows,cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i],fontsize = 16)
        plt.imshow(cvtRGB(img[i]),interpolation=None if interp else 'None')

def plot_sample_images():
	images,labels  = next(train_batches)
	plot(images,labels)


def train_model(model,X_train,X_test,y_train,y_test,batch_size = 32,num_epochs = 50):


	train_batches = ImageDataGenerator().flow(X_train,y_train,batch_size = batch_size)
	val_batches = ImageDataGenerator().flow(X_test,y_test,batch_size = batch_size)

	model.fit_generator(train_batches,steps_per_epoch=len(X_train)//batch_size,validation_steps=len(X_test)//batch_size,validation_data=val_batches,epochs=num_epochs,verbose=1)
	#plot_history('model',model.history.history,32)
	
	return model

def plot_history(model_name, history, epochs):
    print(model_name)
    plt.figure(figsize=(15, 5))

    # summarize history for accuracy
    plt.subplot(1, 2 ,1)
    plt.plot(np.arange(0, len(history['acc'])), history['acc'], 'r')
    plt.plot(np.arange(1, len(history['val_acc'])+1), history['val_acc'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Accuracy vs. Validation Accuracy')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history['loss'])+1), history['loss'], 'r')
    plt.plot(np.arange(1, len(history['val_loss'])+1), history['val_loss'], 'g')
    plt.xticks(np.arange(0, epochs+1, epochs/10))
    plt.title('Training Loss vs. Validation Loss')
    plt.xlabel('Num of Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='best')


    plt.show()


X_train,X_test,y_train,y_test = get_dataset()

model = get_model(num_classes = 5)

model = train_model(model, X_train, X_test, y_train, y_test,batch_size = 32,num_epochs = 50)

save_model(model)
