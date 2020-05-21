import numpy as np

import sys


def one_hot_encoding(Y):
    # Test using
    # one_hot_encoding([7,9])
    # Output -
    # array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    #       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

    # We are assuming that the number of classes is going to be 10
    number_of_classes = 10

    # initialize Y as empty array
    Y_encoded = np.zeros((len(Y), number_of_classes))

    # Setting the relevant value to 1 to convert to one_hot_encoded vector
    for i in range(len(Y)):
        Y_encoded[i][Y[i]] = 1

    return Y_encoded


def split_data(filename):
    data = []

    data_file = open(filename, "r")

    Y = []

    X = []

    for line in data_file:
        Y.append(int(line.split(',')[0]))

        X.append([float(x) for x in line.split(',')[1:]])

    # Converting to numpy array
    X = np.array(X)

    # Adding 1 before every element
    X = np.insert(X, 0, 1, axis=1)

    Y = one_hot_encoding(np.array(Y))

    return X, Y


def initialize_data(type_of_initialization, hidden_units, file_train_input):
    # X, Y = split_data("tinyTrain.csv")
    X, Y = split_data(file_train_input)

    alpha = np.array(0)

    beta = np.array(0)

    if type_of_initialization == 1:

        # Dimensions are (Dim * (M+1))
        # As we have already added 1 to X, we can directly take X.shape[1]
        alpha = np.random.uniform(low=-0.1, high=0.1, size=(hidden_units, X.shape[1] - 1))

        alpha = np.insert(alpha, 0, 0, axis=1)

        # Dimensions are (K * (D+1))
        # For beta, we have done hidden_units+1 for the Y-dimension as Z0 = 1 i.e adding
        # a coefficient for the bias
        # K is the number of classes
        beta = np.random.uniform(low=-0.1, high=0.1, size=(Y.shape[1], hidden_units))

        beta = np.insert(beta, 0, 0, axis=1)
        # Therefore, the extra column in each of the matrices stores the bias.
        # Value of bias is initialized to 0

    else:

        alpha = np.zeros((hidden_units, X.shape[1]))

        beta = np.zeros((Y.shape[1], hidden_units + 1))

    return X, Y, alpha, beta


# Forward Propagation functions

def linear_forward(value, weight):
    return np.matmul(value, weight.T)


def sigmoid_forward(value):
    return 1 / (1 + np.exp(-value))


def softmax_forward(trial):
    # return np.array([[np.exp(value)/np.sum(np.exp(row)) for value in row] for row in trial])
    return np.array([np.exp(value) / np.sum(np.exp(trial)) for value in trial])


def cross_entropy_forward(y_predicted, y_actual):
    # return np.sum(y_actual*np.log(y_predicted))/ y_predicted.shape[0]
    return - np.sum(y_actual * np.log(y_predicted))


def NNForward_Prop(X, Y, alpha, beta):

    # print("x shape")
    # print(X.shape)

    # print("alpha shape")
    # print(alpha.shape)

    a = linear_forward(X, alpha)

    # print("a shape")
    # print(a.shape)

    z = sigmoid_forward(a)

    # Adding the bias term before sending to the softmax layer
    # z = np.insert(z, 0, 1, axis = 1)
    z = np.append([1], z)

    # print("z shape")
    # print(z.shape)

    b = linear_forward(z, beta)

    # print("Beta shape")
    # print(beta.shape)

    # print("b shape")
    # print(b)

    y_predicted = softmax_forward(b)

    # print("y predicted shape")
    # print(y_predicted)
    # print("Actual y shape")
    # print(Y.shape)

    J = cross_entropy_forward(y_predicted, Y)
    # print(J)

    return X, a, z, b, y_predicted, J


def cross_entropy_backward(y, y_predicted, J, gj):
    gy = y / y_predicted

    return -gy * gj


def softmax_backward(b, y_predicted, gy):
    print(softmax_forward(b))

    return gy * softmax_forward(b) * (1 - softmax_forward(b))


def softmax_sigmoid_backward(y, y_predicted):
    return y_predicted - y


def linear_backward(z, value, g_value, weight):
    # Function call
    # g_beta, gz = linear_backward(z, b, gb, beta)

    # Converting the lists to arrays for matrix multiplication
    z = np.array([z])

    # print(g_value)
    # print(z)

    # Converting the lists to arrays for matrix multiplication
    # This will be gALPHA or gBETA
    g_weight = np.matmul(np.array([g_value]).T, z)

    # print("gBETA/gBETA")
    # print(g_weight.shape)

    g_value_derivative = np.matmul(weight.T, g_value)

    # Truncating the bias term
    g_value_derivative = g_value_derivative[1:]

    # print("gb / gx")
    # print(g_value_derivative.shape)

    return g_weight, g_value_derivative


def sigmoid_backward(a, z, gz):
    return gz * sigmoid_forward(a) * (1 - sigmoid_forward(a))


def NNBackward_prop(x, y, alpha, beta, a, z, b, y_predicted, J):
    # This is the derivative of J wrt J therefore 1
    gj = 1

    gy = cross_entropy_backward(y, y_predicted, J, gj)

    # print("gy")
    # print(gy)
    # print("Softmax backward ")
    # gb = softmax_backward(b, y_predicted, gy)

    gb = softmax_sigmoid_backward(y, y_predicted)
    # print("d(loss)/d(softmax inputs)")
    # print(gb)

    g_beta, gz = linear_backward(z, b, gb, beta)

    ga = sigmoid_backward(a, z, gz)

    g_alpha, gx = linear_backward(x, b, ga, alpha)

    return g_alpha, g_beta


# Pass in the required test, train filenames
def Stochastic_Gradient_Descent(file_train_input, file_test_input, training_label_filename, testing_label_filename, metrics_output_filename, num_epochs,num_hidden_units, initialization_flag, learning_rate_input):
    # Used to select zero and random initialization
    # Commandline input
    # 1 = random initialization with uniform distribution between [-0.1, 0.1]
    # 2 = zero initialization
    type_of_initialization_flag = initialization_flag

    # Learning rate for gradient descent
    # Commandline input
    learning_rate = learning_rate_input

    # Hidden units
    # Commandline input
    hidden_units = num_hidden_units

    # Number of epochs
    # Commandline input
    epochs = num_epochs


    # Opening file to write the output
    fileWrite = open(metrics_output_filename, "w+")

    # Initializing the values
    X_train, Y_train, alpha, beta = initialize_data(type_of_initialization_flag, hidden_units, file_train_input)

    # X_test, Y_test = split_data("tinyTest.csv")
    X_test, Y_test = split_data(file_test_input)

    # Traversing over all the number of epochs
    for epoch in range(1, epochs + 1):

        # print("Cost for epoch: "+ str(epoch))

        # Traversing over the number of examples
        for row in range(0, X_train.shape[0]):
            # FORWARD PROPAGATION
            x, a, z, b, y_predicted, J = NNForward_Prop(X_train[row], Y_train[row], alpha, beta)

            # Forward Propagation output
            # print(y_predicted)
            # print("Row:" + str(row) + "- cost: " + str(J))

            # BACKWARD PROPAGATION
            # Will return the gradient values for both the
            g_alpha, g_beta = NNBackward_prop(X_train[row], Y_train[row], alpha, beta, a, z, b, y_predicted, J)

            alpha = alpha - learning_rate * (g_alpha)

            beta = beta - learning_rate * (g_beta)

            # print("Alpha")
            # print(alpha)

            # print("Beta")
            # print(beta)

        training_result_labels, cross_entropy_training_error = calculate_cross_entropy_error(X_train, Y_train, alpha,
                                                                                                beta)
        #print("epoch=" + str(epoch) + " crossentropy(train): " + str(cross_entropy_training_error))

        fileWrite.write("epoch=" + str(epoch) + " crossentropy(train): " + str(cross_entropy_training_error)+"\n")

        testing_result_labels, cross_entropy_testing_error = calculate_cross_entropy_error(X_test, Y_test, alpha,
                                                                                              beta)
        #print("epoch=" + str(epoch) + " crossentropy(test): " + str(cross_entropy_testing_error))

        fileWrite.write("epoch=" + str(epoch) + " crossentropy(test): " + str(cross_entropy_testing_error)+"\n")

    #print("error(train): " + str(calculate_error(X_train, Y_train, alpha, beta)))

    fileWrite.write("error(train): " + str(calculate_error(X_train, Y_train, alpha, beta))+"\n")

    #print("error(test): " + str(calculate_error(X_test, Y_test, alpha, beta)))

    fileWrite.write("error(test): " + str(calculate_error(X_test, Y_test, alpha, beta))+"\n")

    fileWrite.close()

    write_output_to_file(training_label_filename, training_result_labels)

    write_output_to_file(testing_label_filename, testing_result_labels)



def write_output_to_file(file_to_create, data_to_write):

    index = 0

    with open(file_to_create, 'w+') as filehandle:

        for label in data_to_write:

            filehandle.write(str(label))

            if(index != (len(data_to_write)-1)):

                filehandle.write('\n')

            index = index + 1


def cross_entropy_forward_matrix(y_predicted, y_actual):
    return np.sum(y_actual * np.log(y_predicted)) / len(y_predicted)
    # return - np.sum(y_actual*np.log(y_predicted))


def softmax_forward_matrix(trial):
    return np.array([[np.exp(value) / np.sum(np.exp(row)) for value in row] for row in trial])
    # return np.array([np.exp(value)/np.sum(np.exp(trial)) for value in trial])


def calculate_cross_entropy_error(X, Y, alpha, beta):
    a = linear_forward(X, alpha)

    z = sigmoid_forward(a)

    # Adding the bias term before sending to the softmax layer
    z = np.insert(z, 0, 1, axis=1)

    b = linear_forward(z, beta)

    # print(b)

    y_predicted = softmax_forward_matrix(b)

    # print(y_predicted)

    cross_entropy_error = - cross_entropy_forward_matrix(y_predicted, Y)

    predicted_labels = []

    for row in range(0, len(y_predicted)):
        predicted_labels.append(np.argmax(y_predicted[row]))

    return predicted_labels, cross_entropy_error


def calculate_error(X, Y, alpha, beta):
    a = linear_forward(X, alpha)

    z = sigmoid_forward(a)

    # Adding the bias term before sending to the softmax layer
    z = np.insert(z, 0, 1, axis=1)
    # z = np.append([1], z)

    # print("z shape")
    # print(z.shape)

    b = linear_forward(z, beta)

    # print("b")
    # print(b)

    y_predicted = softmax_forward(b)

    error = 0

    for row in range(0, len(y_predicted)):

        if (np.argmax(y_predicted[row]) != np.argmax(Y[row])):
            error += 1

    return error / len(y_predicted)




if __name__ == "__main__":

    file_train_input = sys.argv[1]

    file_test_input = sys.argv[2]

    training_label_filename = sys.argv[3]

    testing_label_filename = sys.argv[4]

    metrics_output_filename = sys.argv[5]

    num_epochs = int(sys.argv[6])

    num_hidden_units = int(sys.argv[7])

    initialization_flag = int(sys.argv[8])

    learning_rate = float(sys.argv[9])

    Stochastic_Gradient_Descent(file_train_input, file_test_input, training_label_filename, testing_label_filename, metrics_output_filename, num_epochs,num_hidden_units, initialization_flag, learning_rate)







