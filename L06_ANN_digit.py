import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
import csv
import argparse

parser = argparse.ArgumentParser("digit analyser")
parser.add_argument("print_explicit", help="If 2, will do all print statements. If 1, prints excluding batches. If 0, no prints.", choices=["0","1","2"])
parser.add_argument("sampleQuantity", help="The number of random samples to generate from the parameter possibilities", type=int)
args = parser.parse_args()
val_print_explicit = int(args.print_explicit)
sampleQuantity = int(args.sampleQuantity)


### Custom class for results and model parameters

#This isnt all of the activation functions, but they are at least the ones i remember from class
_ACTIVATION_FUNCTIONS = {
    "relu":AF.relu,
    "silu":AF.silu,
    "sigmoid":AF.sigmoid,
}

class _model_results:
    def updateResults(self, training_time, accuracy):
        self.training_time= training_time
        self.accuracy= accuracy

class MLP_model_results(_model_results):
    def __init__(self, epochs:int, number_of_layers:int, number_of_hidden_neurons:int, mini_batch_size:int, activationFunction:str, loss_Function:str, gradient_method:str, alpha_learning_rate:float=0.9, gamma_momentum:float=0.5, rho:float=0.9, dropout:float=0.05, training_methods:list[str]=[]) -> None:
        self.epochs = epochs
        self.number_of_layers = number_of_layers
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.mini_batch_size = mini_batch_size
        self.activationFunction = activationFunction
        self.loss_Function = loss_Function
        self.grad_method = gradient_method
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma_momentum = gamma_momentum
        self.rho = rho
        self.dropout = dropout
        self.training_methods = training_methods
        self.training_time = 0.0
        self.accuracy = 0.0

    def __str__(self):
        return f'''MLP Model Results:
Number of Hidden Layers: {self.number_of_layers}
Number of Neurons: {self.number_of_hidden_neurons}
Mini Batch Size: {self.mini_batch_size}
Activation Function: {self.activationFunction}
Loss Function: {self.loss_Function}
Gradient Method: {self.grad_method}
Learning Rate (Alpha): {self.alpha_learning_rate}
Momentum (Gamma): {self.gamma_momentum}
Dropout: {self.dropout}
Training Methods: {", ".join(self.training_methods)}
Training Time: {self.training_time}
Accuracy: {self.accuracy}'''
    
    def to_csv(self):
        return[self.epochs,self.number_of_layers,self.number_of_hidden_neurons,self.mini_batch_size,
               self.activationFunction,self.loss_Function,self.grad_method,self.alpha_learning_rate,self.gamma_momentum,
               self.dropout,self.training_time,self.accuracy]



###################### Designing an ANN architectures #########################

### MLP architecture
class MLP(nn.Module): # All models should inherit from nn.Module
    # This part can be changed based on the design decision.
    def __init__(self, num_input, hidden1_size, num_classes): # Define our ANN structures here
        super(MLP, self).__init__()
        # nn.Linear(in_features, out_features, bias): y = w^Tx + bias
        self.hidden1 = nn.Linear(num_input, hidden1_size)    # connection between input and hidden layer1
        self.output = nn.Linear(hidden1_size, num_classes)
        
        # The model structure can be also defined using "sequential" function
        # self.seq_linear=nn.Sequential(nn.Linear(num_input, hidden1_size),nn.RELU(),nn.Linear(hidden1_size, num_classes))

    # Define "forward" function to perform the computation for input x and return output(s).
    # The function name "forward" is required by Pytorch.
    def forward(self, x):
        # In this implementation, the activation function is reLU, but you can try other functions
        # torch.nn.functional modeule consists of all the activation functions and output functions
        h1_out = _ACTIVATION_FUNCTIONS[iterMLPTestingModel.activationFunction](self.hidden1(x))
        # h1_out = AF.relu(self.hidden1(x))
        output = self.output(h1_out)
        # AF.softmax() is NOT needed when CrossEntropyLoss() is used as it already combines both LogSoftMax() and NLLLoss()
        
        # return self.seq_linear(x) # If the model structrue is define by sequential function.
        return output

### CNN architecture
class CNN(nn.Module):
    # The probability of dropout, number of hidden nodes, number of output classes
    def __init__(self, dropout_pr, num_hidden, num_classes):
        super(CNN, self).__init__()
        # Conv2d is for two dimensional convolution (which means input image is 2D)
        # Conv2d(in_channels=, out_channels=, kernel_size=, stride=1, padding=0)
        # in_channels=1 if grayscale, 3 for color; out_channels is # of output channels (or # of kernels)
        # Ouput size for W from CONV = (W-F+2P)/S+1 where W=input_size, F=filter_size, S=stride, P=padding
        # max_pool2d(input_tensor, kernel_size, stride=None, padding=0) for 2D max pooling
        # Output size for W from POOL = (W-F)/S+1 where S=stride=dimension of pool
        # K is # of channels for convolution layer; D is # of channels for pooling layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # K=D=10, output_size W=(28-5)/1+1=24 (24x24), (default S=1)
        self.pool1 = nn.MaxPool2d(4, 4) # W = (24-4)/4+1=6 (6x6), S=4 (pool dimension) since no overlapping regions
        self.dropout_conv1 = nn.Dropout2d(dropout_pr) # to avoid overfitting by dropping some nodes
        #+ You can add more convolutional and pooling layers
        # Fully connected layer after convolutional and pooling layers
        self.num_flatten_nodes = 10*6*6 # Flatten nodes from 10 channels and 6*6 pool_size = 10*6*6=360
        self.fc1 = nn.Linear(self.num_flatten_nodes, num_hidden)
        #+ You can add more hidden layers here if necessary
        self.out = nn.Linear(num_hidden, num_classes) # the output nodes are 10 classes (10 digits)
        
    def forward(self, x):
        out = AF.relu(self.pool1(self.conv1(x)))
        out = AF.relu(self.dropout_conv1(out))
        out = out.view(-1, self.num_flatten_nodes) # flattening
        out = AF.relu(self.fc1(out))
        # Apply dropout for the randomly selected nodes by zeroing out before output during training
        out = AF.dropout(out)
        output = self.out(out)
        return output

# To display some images
def show_some_digit_images(images):
    if val_print_explicit > 0:
        print("> Shapes of image:", images.shape)
        #print("Matrix for one image:")
        #print(images[1][0])
        # for i in range(0, 10):
        #     plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        #     plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
        # plt.show()

# Training function
def train_ANN_model(num_epochs, training_data, device, CUDA_enabled, is_MLP, ANN_model, loss_func, optimizer):
    train_losses = []
    ANN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
    for epoch_cnt in range(num_epochs):
        for batch_cnt, (images, labels) in enumerate(training_data):
            # Each batch contain batch_size (100) images, each of which 1 channel 28x28
            # print(images.shape) # the shape of images=[100,1,28,28]
            # So, we need to flatten the images into 28*28=784
            # -1 tells NumPy to flatten to 1D (784 pixels as input) for batch_size images
            if (is_MLP):
                # the size -1 is inferred from other dimensions
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)

            optimizer.zero_grad() # set the cumulated gradient to zero
            output = ANN_model(images) # feedforward images as input to the network
            loss = loss_func(output, labels) # computing loss
            #print("Loss: ", loss)
            #print("Loss item: ", loss.item())
            train_losses.append(loss.item())
            # PyTorch's Autograd engine (automatic differential (chain rule) package) 
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation

            # Display the training status
            if (batch_cnt+1) % mini_batch_size == 0:
                if val_print_explicit > 1:
                    print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
    return train_losses

# Testing function
def test_ANN_model(device, CUDA_enabled, is_MLP, ANN_model, testing_data):
    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients    
    predicted_digits=[]
    # torch.no_grad() deactivates Autogra engine (for weight updates). This help run faster
    with torch.no_grad():
        ANN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (is_MLP):
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = ANN_model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction==labels).sum().item()
            accuracy = num_correct/num_samples
            if (batch_cnt+1) % mini_batch_size == 0:
                if val_print_explicit > 1:
                    print(f"batch={batch_cnt+1}/{num_test_batches}")
        if val_print_explicit > 0:
            print("> Number of samples=", num_samples, "number of correct prediction=", num_correct, "accuracy=", accuracy)
        
    return predicted_digits,accuracy

########################### Checking GPU and setup #########################
### CUDA is a parallel computing platform and toolkit developed by NVIDIA. 
# CUDA enables parallelize the computing intensive operations using GPUs.
# In order to use CUDA, your computer needs to have a CUDA supported GPU and install the CUDA Toolkit
# Steps to verify and setup Pytorch and CUDA Toolkit to utilize your GPU in your machine:
# (1) Check if your computer has a compatible GPU at https://developer.nvidia.com/cuda-gpus
# (2) If you have a GPU, continue to the next step, else you can only use CPU and ignore the rest steps.
# (3) Downloaded the compatible Pytorch version and CUDA version, refer to https://pytorch.org/get-started/locally/
# Note: If Pytorch and CUDA versions are not compatible, Pytorch will not be able to recognize your GPU
# (4) The following codes will verify if Pytorch is able to recognize the CUDA Toolkit:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    # Device configuration: use GPU if available, or use CPU
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    print("GPU will be utilized for computation.")
else:
    print("CUDA is supported in your machine. Only CPU will be used for computation.")
#exit()

############################### ANN modeling #################################
### Convert the image into numbers: transforms.ToTensor()
# It separate the image into three color channels RGB and converts the pixels of each images to the brightness
# of the color in the range [0,255] that are scaled down to a range [0,1]. The image is now a Torch Tensor (array object)
### Normalize the tensor: transforms.Normalize() normalizes the tensor with mean (0.5) and stdev (0.5)
#+ You can change the mean and stdev values
print("------------------ANN modeling---------------------------")

#
# Parameters for tweaking
#
testingModels = {"MLP":[],"CNN":[]}
testingResults = {"MLP":[],"CNN":[]}

testingResultsFileDistiction = "HiddenLayerNeurons"
testingResultsFileName = f"MLP_testing_results-{testingResultsFileDistiction}.csv"

# I think we should random sample from this, instead of nested loops
test_parameters_mlp = { 
                        "epochs":[1,3,5,7,9],
                       "layers": [1,3,5,7,9,15,30,50],
                       "hidden_neurons":[10,25,50,100,200,300],
                        "mini_batch_size":[300,200,100,50,25,10,5],
                        "activationFunction":["relu","silu","sigmoid"],
                        "loss_Function":["crossEntropy"],
                        "grad_method":["SGD","Adadelta","Adagrad"],
                        "alpha_learning_rate":[0.01,0.05,0.1,0.25,0.5,0.75,0.9],
                        "gamma_momentum":[0.01,0.05,0.1,0.25,0.5],
                        "rho":[0.01,0.05,0.1,0.25,0.5,0.75],
                        "dropout":[0.001,0.005,0.01,0.05,0.1],
                        "training_methods":[]
}
# 

testingResultsFileDistiction = "RandomSampling-1"
testingResultsFileName = f"MLP_testing_results-{testingResultsFileDistiction}.csv"

# sampleQuantity = 1

import random

# Function to sample random parameters
def sample_random_parameters(param_dict):
    sampled_params = {}
    for key, value in param_dict.items():
        if len(value) > 0:
            sampled_params[key] = random.choice(value)
        else:
            sampled_params[key] = value
    return sampled_params

for _ in range(sampleQuantity):
    random_parameters = sample_random_parameters(test_parameters_mlp)
    testingModels["MLP"].append(MLP_model_results(
        epochs=random_parameters['epochs'],
        number_of_layers=random_parameters['layers'],
        number_of_hidden_neurons=random_parameters['hidden_neurons'],
        mini_batch_size=random_parameters['mini_batch_size'],
        activationFunction=random_parameters["activationFunction"],
        loss_Function=random_parameters['loss_Function'],
        gradient_method=random_parameters['grad_method'],
        alpha_learning_rate=random_parameters['alpha_learning_rate'],
        gamma_momentum=random_parameters['gamma_momentum'],
        rho=random_parameters['rho'],
        dropout=random_parameters['dropout'],
        training_methods=[]
    ))
# Sample a set of random parameters
if val_print_explicit > 0:
    print(testingModels)

# Pretty good results for these
testingModels["MLP"].append(MLP_model_results(
    epochs = 3,
    number_of_layers=3,
    number_of_hidden_neurons=25,
    mini_batch_size=25,
    activationFunction="sigmoid",
    loss_Function="crossEntropy",
    gradient_method="Adadelta",
    alpha_learning_rate=0.5 ,
    gamma_momentum=0.05,
    dropout=0.01,
    training_methods=[]
)
)

# End Tweaking Parameters

for iterMLPTestingModel in testingModels["MLP"]:
    if not isinstance(iterMLPTestingModel,MLP_model_results):
        continue


    transforms_result = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
    # PyTorch tensors are like NumPy arrays that can run on GPU
    # e.g., x = torch.randn(64,100).type(dtype) # need to cast tensor to a CUDA datatype (dtype)

    from torch.autograd import Variable
    x = Variable

    ### Download and load the dataset from the torch vision library to the directory specified by root=''
    # MNIST is a collection of 7000 handwritten digits (in images) split into 60000 training images and 1000 for testing 
    # PyTorch library provides a clean data set. The following command will download training data in directory './data'
    train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms_result, download=True)
    test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms_result, download=False)
    if val_print_explicit > 0:
        print("> Shape of training data:", train_dataset.data.shape)
        print("> Shape of testing data:", test_dataset.data.shape)
        print("> Classes:", train_dataset.classes)

    # You can use random_split function to splite a dataset
    #from torch.utils.data.dataset import random_split
    #train_data, val_data, test_data = random_split(train_dataset, [60,20,20])

    ### DataLoader will shuffle the training dataset and load the training and test dataset

    mini_batch_size = iterMLPTestingModel.mini_batch_size #+ You can change this mini_batch_size

    # If mini_batch_size==100, # of training batches=6000/100=600 batches, each batch contains 100 samples (images, labels)
    # DataLoader will load the data set, shuffle it, and partition it into a set of samples specified by mini_batch_size.
    train_dataloader=DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
    test_dataloader=DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=True)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)
    if val_print_explicit > 0:
        print("> Mini batch size: ", mini_batch_size)
        print("> Number of batches loaded for training: ", num_train_batches)
        print("> Number of batches loaded for testing: ", num_test_batches)

    ### Let's display some images from the first batch to see what actual digit images look like
    iterable_batches = iter(train_dataloader) # making a dataset iterable
    images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
    show_digit_image = True
    if show_digit_image:
        show_some_digit_images(images)

    ### Create an object for the ANN model defined in the MLP class
    # Architectural parameters: You can change these parameters except for num_input and num_classes
    num_input = 28*28   # 28X28=784 pixels of image
    num_classes = 10    # output layer
    #There are 10 digits that we are trying to classify

    num_hidden = iterMLPTestingModel.number_of_hidden_neurons
    # num_hidden = 10     # number of neurons at the first hidden layer
    # Randomly selected neurons by dropout_pr probability will be dropped (zeroed out) for regularization.

    dropout_pr = iterMLPTestingModel.dropout
    # dropout_pr = 0.05

    # MLP model
    MLP_model=MLP(num_input, num_hidden, num_classes)
    # Some model properties: 
    # .state_dic(): a dictionary of trainable parameters with their current valeus
    # .parameter(): a list of all trainable parameters in the model
    # .train() or .eval(): setting training, testing mode
    if val_print_explicit > 0:
        print("> MLP model parameters")
        print(MLP_model.parameters)
        # state_dict() maps each layer to its parameter tensor.
        print ("> MLP model's state dictionary")
        for param_tensor in MLP_model.state_dict():
            print(param_tensor, MLP_model.state_dict()[param_tensor].size())

    #exit()

    # To turn on/off CUDA if I don't want to use it.
    CUDA_enabled = True
    if (device.type == 'cuda' and CUDA_enabled):
        print("...Modeling MLP using GPU...")
        MLP_model = MLP_model.to(device=device) # sending to whaever device (for GPU acceleration)
        # CNN_model = CNN_model.to(device=device)
    else:
        print("...Modeling MLP using CPU...")



    ### Choose a gradient method
    # model hyperparameters and gradient methods
    # optim.SGD performs gradient descent and update the weigths through backpropagation.
    num_epochs = iterMLPTestingModel.epochs
    alpha = iterMLPTestingModel.alpha_learning_rate       # learning rate
    gamma = iterMLPTestingModel.gamma_momentum        # momentum
    rho = iterMLPTestingModel.rho
    # Stochastic Gradient Descent (SGD) is used in this program.
    #+ You can choose other gradient methods (Adagrad, adadelta, Adam, etc.) and parameters
    if iterMLPTestingModel.grad_method == "SGD":
        MLP_optimizer = optim.SGD(MLP_model.parameters(), lr=alpha, momentum=gamma)
    elif iterMLPTestingModel.grad_method == "Adadelta":
        MLP_optimizer = optim.Adadelta(MLP_model.parameters(), lr=alpha, rho=rho)
    elif iterMLPTestingModel.grad_method == "Adagrad":
        MLP_optimizer = optim.Adagrad(MLP_model.parameters(), lr=alpha)
    if val_print_explicit > 0:
        print("> MLP optimizer's state dictionary")
        for var_name in MLP_optimizer.state_dict():
            print(var_name, MLP_optimizer.state_dict()[var_name])

    ### Define a loss function: You can choose other loss functions
    loss_func = nn.CrossEntropyLoss()


    ### Train your networks
    if val_print_explicit > 0:
        print("............Training MLP................")
    is_MLP = True
    
    start_time = time()
    train_loss=train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, is_MLP, MLP_model, loss_func, MLP_optimizer)
    training_time = time() - start_time
    if val_print_explicit > 0:
        print("............Testing MLP model................")

        print("> Input digits:")
        print(labels)
    predicted_digits, accuracy=test_ANN_model(device, CUDA_enabled, is_MLP, MLP_model, test_dataloader)
    if val_print_explicit > 0:
        print("> Predicted digits by MLP model")
        print(predicted_digits)
    iterMLPTestingModel.updateResults(training_time=training_time,accuracy=accuracy)
    testingResults["MLP"].append(iterMLPTestingModel)

if val_print_explicit > 0:
    print("All MLP tests completed:")

# with open(testingResultsFileName,"w",newline="") as file:
with open("/mountedStorage/" + testingResultsFileName,"w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epochs","number_of_layers","number_of_hidden_neurons","mini_batch_size",
            "activationFunction","loss_Function","grad_method","alpha_learning_rate","gamma_momentum",
            "dropout","training_time","accuracy"])
    if val_print_explicit > 0:
        for iterMLPresult in testingResults["MLP"]:
            print(iterMLPresult)

            writer.writerow(iterMLPresult.to_csv())


exit()

# CNN model
CNN_model = CNN(dropout_pr, num_hidden, num_classes)
print("> CNN model parameters")
print(CNN_model.parameters)

# To turn on/off CUDA if I don't want to use it.
CUDA_enabled = True
if (device.type == 'cuda' and CUDA_enabled):
    print("...Modeling CNN using GPU...")
    # MLP_model = MLP_model.to(device=device) # sending to whaever device (for GPU acceleration)
    CNN_model = CNN_model.to(device=device)
else:
    print("...Modeling CNN using CPU...")

# CNN optimizer
CNN_optimizer = optim.SGD(CNN_model.parameters(), lr=alpha, momentum=gamma) 


print("............Training CNN................")
is_MLP = False
train_loss=train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, is_MLP, CNN_model, loss_func, CNN_optimizer)
print("............Testing CNN model................")
predicted_digits=test_ANN_model(device, CUDA_enabled, is_MLP, CNN_model, test_dataloader)
print("> Predicted digits by CNN model")
print(predicted_digits)

#### To save and load models and model's parameters ####
# To save and load model parameters
#print("...Saving and loading model states and model parameters...")
#torch.save(MLP_model.state_dict(), 'MLP_model_state_dict.pt')
#loaded_MLP_model=MLP(num_input, num_hidden, num_classes)
#loaded_MLP_model=MLP_model.load_state_dict(torch.load('MLP_model_state_dict.pt'))
#torch.save(MLP_optimizer.state_dict(), 'MLP_optimizer_state_dict.pt')
#loaded_MLP_optimizer = MLP_optimizer.load_state_dict(torch.load('MLP_optimizer_state_dict.pt'))

# To save and load a model
#print("...Saving model...")
#torch.save(MLP_model, 'MLP_model_NNIST.pt')
#pretrained_model = torch.load('MLP_model_NNIST.pt')
