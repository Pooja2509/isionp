# organize imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing import image
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image

from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels


# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch = 25
num_classes = 10
batch_size = 128
train_size = 60000
test_size = 10000
v_length = 784

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
print ("[INFO] train data shape: {}".format(trainData.shape))
print ("[INFO] test data shape: {}".format(testData.shape))
print ("[INFO] train samples: {}".format(trainData.shape[0]))
print ("[INFO] test samples: {}".format(testData.shape[0]))

# reshape the dataset
trainData = trainData.reshape(train_size, v_length)
testData = testData.reshape(test_size, v_length)
trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainData /= 255
testData /= 255


print ("[INFO] train data shape: {}".format(trainData.shape))
print ("[INFO] test data shape: {}".format(testData.shape))
print ("[INFO] train samples: {}".format(trainData.shape[0]))
print ("[INFO] test samples: {}".format(testData.shape[0]))


# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels = np_utils.to_categorical(testLabels, num_classes)

# create the model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# summarize the model
model.summary()

# compile the model
model.compile(loss="categorical_crossentropy",
			  optimizer="adam",
			  metrics=["accuracy"])


# fit the model
history = model.fit(trainData,
				 	mTrainLabels,
					validation_data=(testData, mTestLabels),
					batch_size=batch_size,
					nb_epoch=nb_epoch,
					verbose=2)

# print the history keys
print (history.history.keys())


# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# history plot for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# history plot for accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# print the results
print("[INFO] test score - {}".format(scores[0]))
print("[INFO] test accuracy - {}".format(scores[1]))
with open('E:\isionp\model_architecture1.json', 'w') as f:
    f.write(model.to_json())

# Save the weights
model.save_weights('E:\isionp\weights.h5')

# Save the model architecture

print("All Done")



img = image.load_img(path="E:\isionp\mnist.png\mset1\7",grayscale=True,target_size=(28,28,1))
img = image.img_to_array(img)
test_img = img.reshape((1,784))

prediction = model.predict_classes(test_img)
print("[INFO] I think the digit is - {}".format(prediction))

#Prediction Test

# grab some test images from the test data
#test_images = testData[1:5]

# reshape the test images to standard 28x28 format
#test_images = test_images.reshape(test_images.shape[0], 28, 28)

# loop over each of the test images
#for i, test_image in enumerate(test_images, start=1):
	# grab a copy of test image for viewing
#	org_image = test_image

	# reshape the test image to [1x784] format so that our model understands
#	test_image = test_image.reshape(1, 784)

	# make prediction on test image using our trained model
#	prediction = model.predict_classes(test_image, verbose=0)

	# display the prediction and image
#	print("[INFO] I think the digit is - {}".format(prediction[0]))