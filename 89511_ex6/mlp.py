import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def init_mlp(inputs, targets, nhidden):
    """ Initialize network """

    # Set up network size
    nin = np.shape(inputs)[1]
    nout = np.shape(targets)[1]
    ndata = np.shape(inputs)[0]
    nhidden = nhidden

    #Initialize network
    weights1 = (np.random.rand(nin+1, nhidden)-0.5)*2/np.sqrt(nin)
    weights2 = (np.random.rand(nhidden+1, nout)-0.5)*2/np.sqrt(nhidden)

    return weights1, weights2


def loss_and_gradients(input_x, expected_output_y, weights1, weights2):
    """compute loss and gradients for a given x,y
    
    this function gets an (x,y) pair as input along with the weights of the mlp,
    computes the loss on the given (x,y), computes the gradients for each weights layer,
    and returns a tuple of loss, weights 1 gradient, weights 2 gradient.
    The loss should be calculated according to the loss function presented in the assignment
    
    Arguments:
        input_x {numpy 1d array} -- an instance from the dataset
        expected_output_y {scalar} -- the ground truth
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    
    Returns:
        tuple -- loss, weights 1 gradient, weights 2 gradient, and activations[-1] which is y_hat
    """
    # Initialize gradients
    weights1_gradient, weights2_gradient = np.zeros(weights1.shape), np.zeros(weights2.shape)
    # Initialize loss
    loss = 0
    weighted_outputs, activations = mlpfwd(input_x, weights1, weights2)

    #**************************YOUR CODE HERE*********************
    #*************************************************************
    #Write the backpropagation algorithm to find the update values for weights1 and weights2.
    ones = np.ones(len(activations[-1]))
    hidden_activation = np.append(activations[:-1], 1)
    delta_0 = np.subtract(activations[-1], expected_output_y) * activations[-1] * (ones - activations[-1])
    for k in range(weights2.shape[1]):
        delta_0_k = delta_0[k]
        for j in range(weights2.shape[0]):
            weights2_gradient[j][k] = delta_0_k * hidden_activation[j]


    for j in range(weights1.shape[1]):
        sum_output = 0
        for k in range(weights2.shape[1]):
            delta_0_k = delta_0[k]
            sum_output += delta_0_k * weights2[j][k]
        delta_h_j = activations[0][j] * (1 - activations[0][j]) * sum_output
        for i in range(weights1.shape[0]):
            weights1_gradient[i][j] = delta_h_j * input_x[i]

    for k in range(weights2.shape[1]):
        loss += 0.5 * (activations[-1][k] - expected_output_y[k])**2

    #*************************************************************
    #*************************************************************

    return loss, weights1_gradient, weights2_gradient, activations[-1]


def mlpfwd(input_x, weights1, weights2):
    """feed forward
    
    this function gets an input x and feeds it through the mlp.
    
    Arguments:
        input_x {numpy 1d array} -- an instance from the dataset
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    
    Returns:
        tuple -- list of weighted outputs along the way, list of activations along the way:
        
        1) The first part of the tuple consists of a list, where every item in the list
        holds the values of a layer in the network, before the activation function has been applied
        on it. The value of a layer in the network is the weighted sum of the layer before it.
        
        2) The second part of the tuple consists of a list, where every item in the list holds
        the values of a layer in the network, after the activation function has been applied on it.
        Don't forget to add the bias to a layer, when required.
    """

    weighted_outputs, activations = [], []

    #**************************YOUR CODE HERE*********************
    #*************************************************************
    nhidden = weights1.shape[1]

    hidden_activations = []
    # hidden layer outputs
    for i in range(nhidden):
        layer_output = np.sum(np.dot(input_x, weights1[:, i]))
        weighted_outputs.append(layer_output)
        hidden_activations.append(sigmoid(layer_output))
    activations.append(hidden_activations)

    # output layer
    output_activations = []
    input_hidden = np.append(np.asarray(activations), [1])
    noutput = weights2.shape[1]
    for i in range(noutput):
        layer_output = np.sum(np.dot(input_hidden, weights2[:, i]))
        weighted_outputs.append(layer_output)
        output_activations.append(sigmoid(layer_output))
    activations.append(output_activations)
    #*************************************************************
    #*************************************************************


    return weighted_outputs, activations


def accuracy_on_dataset(inputs, targets, weights1, weights2):
    """compute accuracy
    
    this function gets a dataset and returns model's accuracy on the dataset.
    The accuracy is calculated using a threshold of 0.5:
    if the prediction is >= 0.5 => y_hat = 1
    if the prediction is < 0.5 => y_hat = 0
    
    Arguments:
        inputs {numpy 2d array} -- instances
        targets {numpy 2d array} -- ground truths
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer

    Returns:
        scalar -- accuracy on the given dataset
    """

    #**************************YOUR CODE HERE*********************
    #*************************************************************
    correct_predictions = 0.0
    nsamples = inputs.shape[0]
    for i in range(nsamples):
        weighted_outputs, activations = mlpfwd(inputs[i], weights1, weights2)

        y_hat = np.zeros(len(activations[-1]))
        for k in range(len(activations[-1])):
            if activations[-1][k] >= 0.5:
                y_hat[k] = 1
            else:
                y_hat[k] = 0

        if np.array_equal(y_hat, targets[i]):
            correct_predictions += 1

    accuracy = correct_predictions/nsamples
    #*************************************************************
    #*************************************************************

    return accuracy


def mlptrain(inputs, targets, eta, nepochs, weights1, weights2):
    """train the model
    
    Arguments:
        inputs {numpy 2d array} -- instances
        targets {numpy 2d array} -- ground truths
        eta {scalar} -- learning rate
        nepochs {scalar} -- number of epochs
        weights1 {numpy 2d array} -- weights between input layer and hidden layer
        weights2 {numpy 2d array} -- weights between hidden layer and output layer
    """
    ndata = np.shape(inputs)[0]
    # Add the inputs that match the bias node
    inputs = np.concatenate((inputs,np.ones((ndata,1))),axis=1)

    for n in range(nepochs):
        epoch_loss = 0
        predictions = []
        for ex_idx in range(len(inputs)):
            x = inputs[ex_idx]
            y = targets[ex_idx]
            
            # compute gradients and update the mlp
            loss, weights1_gradient, weights2_gradient, y_hat= loss_and_gradients(x, y, weights1, weights2)
            weights1 -= eta * weights1_gradient
            weights2 -= eta * weights2_gradient
            epoch_loss += loss
            predictions.append(y_hat)

        if (np.mod(n,100)==0):
            print n, epoch_loss, accuracy_on_dataset(inputs, targets, weights1, weights2)

    return weights1, weights2
        
        



