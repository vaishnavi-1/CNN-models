# KEYWORDS AND FUNCTIONS USED TO BUILD CNN MODEL USING KERAS

- **Convolution** : A convolution multiplies a matrix of pixels with a filter matrix or ‘kernel’ and sums up the multiplication values. Then the convolution slides over to the next pixel and repeats the same process until all the image pixels have been covered.
- **MNIST Dataset** : This dataset consists of 70,000 images of handwritten digits from 0–9. We will identify them using a CNN model on google collab.
- **one-hot encode** : This means that a column will be created for each output category and a binary variable is inputted for each category. For example, we saw that the first image in the dataset is a 5. This means that the sixth number in our array will have a 1 and the rest of the array will be filled with 0.
- **Sequential** : Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer. Each layer has weights that correspond to the layer the follows it. We use the `.add()` function to add layers to our model. We will add two layers and an output layer.
- **Conv2D layers** : These are convolution layers that will deal with our input images, which are seen as 2-dimensional matrices.
- **kernel size** : It is the size of the filter matrix for our convolution.
- **Activation** : It is the activation function for the layer. The activation function we will be using for our first 2 layers is the ReLU, or Rectified Linear Activation.
- **Flatten layer** : It is used to flatten the input. It is a connection between Convolutional and dense layers.
- **Dense layer** : It is the layer type we will use in for our output layer. Dense is a standard layer type that is used in many cases for neural networks.
- **Softmax** : It makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.
- **Compiling the model** : It takes three parameters.They are:
     - Optimizer : The optimizer controls the learning rate. We will be using `adam` as our optmizer. Adam is generally a good optimizer to use for many cases. The adam optimizer adjusts the learning rate throughout training.
     - Loss : We will use `categorical_crossentropy` for our loss function. A lower score indicates that the model is performing better.
     - Accuracy : We will use the ‘accuracy’ metric to see the accuracy score on the validation set when we train the model.
- `fit()` **function** : The parameters of fit() function are training data (train_X), target data (train_y), validation data, and the number of  epochs.
- `predict()` **function** :  The predict function will give an array with 10 numbers. These numbers are the probabilities that the input image represents each digit (0–9). The array index with the highest number represents the model prediction. The sum of each array equals 1 (since each number is a probability). 
