import matplotlib.pyplot as plt
from get_data import find_category,get_image
import sys
from keras.models import load_model
try:
	img_dir =sys.argv[1]
	img =get_image(path = img_dir)
	model = load_model('Data/Model/Model_save.h5')
	print(f"The Photo is in the category of {find_category(model.predict_classes(img)[0])}")
except:
	print("Sorry!! No File provided for prediction.")





