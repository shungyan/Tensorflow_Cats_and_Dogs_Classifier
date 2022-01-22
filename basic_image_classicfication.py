# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


#give the class name based on the label from 0-9 
#label  class
#0      T-shirt/top
#1      Trouser
#2      Pullover
#3      Dress
#4      Coat
#5      Sandal
#6      Shirt
#7      Sneaker
#8      Bag
#9      Ankle boot
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


fashion_mnist = tf.keras.datasets.fashion_mnist

#split the train and test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#Show the first train image
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''


#turn the pixels from 0-255 to 0-1 because neural network can only take 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0


#plot out the first 25 train images with their labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),#transform 2 dimensional array to 1 dimensional array 28x28=784
    tf.keras.layers.Dense(128, activation='relu'),#hidden layer with 128 nodes
                                                  #activation is relu which means rectified linear unit  
                                                  #will output the input directly if it is positive, else output zero
    tf.keras.layers.Dense(10)#each node contains the score for each class
])


#compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

'''
Loss function — This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
Optimizer — This is how the model is updated based on the data it sees and its loss function.
Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
'''

#fit the model to training data
model.fit(train_images, train_labels, epochs=10)


#evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


#attach a softmax layer to convert logits to probability, easier to intepret
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


#predict the label of the test image
predictions = probability_model.predict(test_images)



#create a function that plot the test images, the predicted label, the confidence level of the predicted label and the true label
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


#create a function that plot the confidence level of predicted label and true label
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#call the functions to test the first test image
  i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)#pass in the prediction label and true label to compare
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)#compare the confidence level of prediction label and true label
plt.show()