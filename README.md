# neural-network-handwritten-letters

This programs assigns labels to handwritten digits using a single layer feed forward back propagation Neural network written from scratch in Python using Numpy. All of the functions needed to initialize, train, evaluate, and make predictions with the network are included. 

In forward propagation, the activation function used is the sigmoid function. In the output layer, softmax function is used to output the final label. Backpropagation is used to update the weights for the network.


Execution - 

python neuralnet.py [args]

Where above [args] is a placeholder for nine command-line arguments described in detail below:
1. train input: path to the training input .csv file
2. test input: path to the test input .csv file 
3. train out: path to output .labels file to which the prediction on the training data should be written
4. test out: path to output .labels file to which the prediction on the test data should be written 
5. metrics out: path of the output .txt file to which metrics such as train and test error should be written 
6. num epoch: integer specifying the number of times backpropogation loops through all of the training data
7. hidden units: positive integer specifying the number of hidden units.
8. init flag: integer taking value 1 or 2 that specifies whether to use RANDOM or ZERO initialization 
9. learning rate: float value specifying the learning rate for SGD.

