from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np

from IPython.display import Image
Image(filename="cat.jpg")

img_path='cat.jpg'
img = image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
model=ResNet50(weights='imagenet')

predict=model.predict(x)
print('predicted',decode_predictions(predict,top=3)[0])

def classify(img_path):
    display(Image(filename=img_path))
    img = image.load_img(img_path,target_size=(224,224))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    predict=model.predict(x)
    print('predicted',decode_predictions(predict,top=3)[0])

classify('download.jpg')