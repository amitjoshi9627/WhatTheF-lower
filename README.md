# Flower_Classifier
### By Amit Joshi

Classifying 5 types of Flowers using Deep Learning.
#### Example:
| <img src="rose.jpg?raw=true" width="200">|<img src="dandelion.jpg?raw=true" width="200">|
|:-:|:-:|
|Rose: 0.986|Dandelion: 0.93|


### Descrition
The dataset is from Tensorflow's [Flowers Recognition]( http://download.tensorflow.org/example_images/flower_photos.tgz \). The goal is to classify five kinds of flowers (daisy, dandelion, roses, sunflowers, tulips) by raw image.

### Dataset
The dataset contains 3670 images of flowers. The pictures are divided into five classes: daisy, dandelion, roses, sunflowers, tulips. For each class there are about 700 photos.

### Preprocessing
1. Resize all the input images to 48x48.
2. 90% training samples && 10% validation samples.

### Model Training
  `python3 train.py`
### Model Predictions
  `python3 predict.py <filename>`
### Notes
* Computing: Google Colab Tesla K80 GPU
* Python version: 3.6.6
* Using packages
  1. [`Keras`](https://www.tensorflow.org/guide/keras) (tensorflow.python.keras) for building models 
  2. [`OpenCV`](https://opencv.org/) (cv2) for processing images
  3. [`sikit-learn`](http://scikit-learn.org/stable/) (sklearn) for train_test_split 
  4. Install necessary modules with `sudo pip3 install -r requirements.txt` command.
