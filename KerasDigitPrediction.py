from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
import PIL
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

# Model reconstruction from JSON file
with open('E:\isionp\model_architecture1.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('E:\isionp\weights.h5')



img = image.load_img(path="E:\isionp\images401.png", grayscale=True, target_size=(28, 28, 1))
# img=mpimg.imread('D:\IoTSolutions\MachineLearning\model\image.PNG')
# imgplot = plt.imshow(img)
# plt.show()


img = image.img_to_array(img)
test_img = img.reshape((1, 784))

prediction1 = model.predict_classes(test_img)
print("[INFO] I think the digit is - {}".format(prediction1))
