from time import time
import csv
import argparse
from model_tools import *

parser = argparse.ArgumentParser("digit analyser")
parser.add_argument("print_explicit", help="If 2, will do all print statements. If 1, prints excluding batches. If 0, no prints.", choices=["0","1","2"])
parser.add_argument("sampleQuantity", help="The number of random samples to generate from the parameter possibilities", type=int)
parser.add_argument("filename", help="The name of the file to save the results to (please do not include the file extension)", type=str)
args = parser.parse_args()
val_print_explicit = int(args.print_explicit)
sampleQuantity = int(args.sampleQuantity)
testingResultsFileDistiction = args.filename


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

class CNN_model_results(_model_results):
    def __init__(self, epochs:int, conv_pool_layers:int, fc_layers:int, conv_kernel_size:int, conv_stride:int, conv_padding:int, pool_kernel_size:int, pool_stride:int, hidden_neurons:int, mini_batch_size:int, activationFunction:str, loss_Function:str, grad_method:str, alpha_learning_rate:float=0.9, gamma_momentum:float=0.5, rho:float=0.9, dropout:float=0.05, training_methods:list[str]=[]) -> None:
        self.epochs = epochs
        self.conv_pool_layers = conv_pool_layers
        self.fc_layers = fc_layers
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.hidden_neurons = hidden_neurons
        self.mini_batch_size = mini_batch_size
        self.activationFunction = activationFunction
        self.loss_Function = loss_Function
        self.grad_method = grad_method
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma_momentum = gamma_momentum
        self.rho = rho
        self.dropout = dropout
        self.training_methods = training_methods
        self.training_time = 0.0
        self.accuracy = 0.0

    def __str__(self):
        return f'''CNN Model Results:
Number of Convolutional and Pooling Layers: {self.conv_pool_layers}
Number of Fully Connected Layers: {self.fc_layers}
Convolutional Kernel Size: {self.conv_kernel_size}
Convolutional Stride: {self.conv_stride}
Convolutional Padding: {self.conv_padding}
Pooling Kernel Size: {self.pool_kernel_size}
Pooling Stride: {self.pool_stride}
Number of Hidden Neurons: {self.hidden_neurons}
Mini Batch Size: {self.mini_batch_size}
Activation Function: {self.activationFunction}
Loss Function: {self.loss_Function}
Gradient Method: {self.grad_method}
Learning Rate (Alpha): {self.alpha_learning_rate}
Momentum (Gamma): {self.gamma_momentum}
Rho: {self.rho}
Dropout: {self.dropout}
Training Methods: {", ".join(self.training_methods)}
Training Time: {self.training_time}
Accuracy: {self.accuracy}'''

    def to_csv(self):
        return[
            self.epochs,
            self.conv_pool_layers,
            self.fc_layers,
            self.conv_kernel_size,
            self.conv_stride,
            self.conv_padding,
            self.pool_kernel_size,
            self.pool_stride,
            self.hidden_neurons,
            self.mini_batch_size,
            self.activationFunction,
            self.loss_Function,
            self.grad_method,
            self.alpha_learning_rate,
            self.gamma_momentum,
            self.rho,
            self.dropout,
            self.training_time,
            self.accuracy
         ]

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

mlptestingResultsFileName = f"MLP_testing_results-{testingResultsFileDistiction}.csv"
cnntestingResultsFileName = f"CNN_testing_results-{testingResultsFileDistiction}.csv"

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

test_parameters_cnn = { 
                        "epochs":[1,3,5,7,9],
                        "conv_pool_layers": [1,2,3,4,5,7,9,15,30,50],
                        "fc_layers": [1,2,3,4,5,7,9,15,30,50],
                        "conv_kernel_size":[3,4,5,6,7],
                        "conv_stride":[1,2,3,4,5],
                        "conv_padding":[0,1,2],
                        "pool_kernel_size":[3,4,5,6],
                        "pool_stride":[1,2,3,4],
                        "hidden_neurons":[5,10,25,50,100,200,300],
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


mlptestingResultsFileName = f"MLP_testing_results-{testingResultsFileDistiction}.csv"
cnntestingResultsFileName = f"CNN_testing_results-{testingResultsFileDistiction}.csv"

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

for _ in range(sampleQuantity):
    random_parameters = sample_random_parameters(test_parameters_cnn)
    testingModels["CNN"].append(CNN_model_results(
        epochs=random_parameters['epochs'],
        conv_pool_layers=random_parameters['conv_pool_layers'],
        fc_layers=random_parameters['fc_layers'],
        conv_kernel_size=random_parameters['conv_kernel_size'],
        conv_stride=random_parameters['conv_stride'],
        conv_padding=random_parameters['conv_padding'],
        pool_kernel_size=random_parameters['pool_kernel_size'],
        pool_stride=random_parameters['pool_stride'],
        hidden_neurons=random_parameters['hidden_neurons'],
        mini_batch_size=random_parameters['mini_batch_size'],
        activationFunction=random_parameters["activationFunction"],
        loss_Function=random_parameters['loss_Function'],
        grad_method=random_parameters['grad_method'],
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
    show_digit_image = False
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

    layers = iterMLPTestingModel.number_of_layers

    activ_func = _ACTIVATION_FUNCTIONS[iterMLPTestingModel.activationFunction]
    l_func = iterMLPTestingModel.loss_Function

    # MLP model
    MLP_model=MLP(num_input, layers, num_hidden, num_classes, dropout_pr, activ_func, l_func)
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

    if l_func == "crossEntropy":
        l_func = nn.CrossEntropyLoss()

    ### Train your networks
    if val_print_explicit > 0:
        print("............Training MLP................")
    is_MLP = True
    
    start_time = time()
    train_loss=train_ANN_model(num_epochs,
                               train_dataloader,
                               device,
                               CUDA_enabled,
                               is_MLP,
                               MLP_model,
                               l_func,
                               MLP_optimizer,
                               mini_batch_size,
                               num_train_batches,
                               val_print_explicit
    )
    training_time = time() - start_time
    if val_print_explicit > 0:
        print("............Testing MLP model................")

        print("> Input digits:")
        print(labels)
    predicted_digits, accuracy=test_ANN_model(device,
                                              CUDA_enabled, 
                                              is_MLP, 
                                              MLP_model, 
                                              test_dataloader,
                                              mini_batch_size,
                                              num_test_batches,
                                              val_print_explicit
    )
    if val_print_explicit > 0:
        print("> Predicted digits by MLP model")
        print(predicted_digits)
    iterMLPTestingModel.updateResults(training_time=training_time,accuracy=accuracy)
    testingResults["MLP"].append(iterMLPTestingModel)

if val_print_explicit > 0:
    print("All MLP tests completed:")

# with open(testingResultsFileName,"w",newline="") as file:
with open(mlptestingResultsFileName,"w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epochs","number_of_layers","number_of_hidden_neurons","mini_batch_size",
            "activationFunction","loss_Function","grad_method","alpha_learning_rate","gamma_momentum",
            "dropout","training_time","accuracy"])
    if val_print_explicit > 0:
        for iterMLPresult in testingResults["MLP"]:
            print(iterMLPresult)

            writer.writerow(iterMLPresult.to_csv())


for iterCNNTestingModel in testingModels["CNN"]:
    if not isinstance(iterCNNTestingModel,CNN_model_results):
        continue

    transforms_result = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
    # PyTorch tensors are like NumPy arrays that can run on GPU
    # e.g., x = torch.randn(64,100).type(dtype) # need to cast tensor to a CUDA datatype (dtype)

    from torch.autograd import Variable
    x = Variable

    ### Download and load the dataset from the torch vision library to the directory specified by root=''
    # MNIST is a collection of 7000 handwritten digits (in images) split into 60000 training images and 1000 for testing 
    # PyTorch library provides a clean data set. The following command will download training data in directory './data'
    train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms_result, download=False)
    test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms_result, download=False)
    if val_print_explicit > 0:
        print("> Shape of training data:", train_dataset.data.shape)
        print("> Shape of testing data:", test_dataset.data.shape)
        print("> Classes:", train_dataset.classes)

    # You can use random_split function to splite a dataset
    #from torch.utils.data.dataset import random_split
    #train_data, val_data, test_data = random_split(train_dataset, [60,20,20])

    ### DataLoader will shuffle the training dataset and load the training and test dataset

    mini_batch_size = iterCNNTestingModel.mini_batch_size #+ You can change this mini_batch_size

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
    show_digit_image = False
    if show_digit_image:
        show_some_digit_images(images)
    
    ### Create an object for the ANN model defined in the MLP class
    # Architectural parameters: You can change these parameters except for num_input and num_classes
    num_input = 28*28   # 28X28=784 pixels of image
    num_classes = 10    # output layer
    #There are 10 digits that we are trying to classify

    num_hidden = iterCNNTestingModel.hidden_neurons

    conv_pool_layers = iterCNNTestingModel.conv_pool_layers
    fc_layers = iterCNNTestingModel.fc_layers
    conv_kernel_size = iterCNNTestingModel.conv_kernel_size
    conv_stride = iterCNNTestingModel.conv_stride
    conv_padding = iterCNNTestingModel.conv_padding
    pool_kernel_size = iterCNNTestingModel.pool_kernel_size
    pool_stride = iterCNNTestingModel.pool_stride

    activ_func = _ACTIVATION_FUNCTIONS[iterCNNTestingModel.activationFunction]
    l_func = iterCNNTestingModel.loss_Function
    grad_method = iterCNNTestingModel.grad_method
    alpha = iterCNNTestingModel.alpha_learning_rate
    gamma = iterCNNTestingModel.gamma_momentum
    rho = iterCNNTestingModel.rho
    dropout_pr = iterCNNTestingModel.dropout

    # CNN model
    CNN_model = CNN(dropout_pr, 
                    num_hidden,
                    num_classes,
                    conv_pool_layers,
                    fc_layers,
                    conv_kernel_size,
                    conv_stride,
                    conv_padding,
                    pool_kernel_size,
                    pool_stride,
                    10,
                    activ_func,
                    l_func
    )
    while(CNN_model.fail):
        print("Failed to create model, trying again")
        random_parameters = sample_random_parameters(test_parameters_cnn)
        iterCNNTestingModel.conv_kernel_size = conv_kernel_size = random_parameters['conv_kernel_size']
        iterCNNTestingModel.conv_stride = conv_stride = random_parameters['conv_stride']
        iterCNNTestingModel.conv_padding = conv_padding = random_parameters['conv_padding']
        iterCNNTestingModel.pool_kernel_size = pool_kernel_size = random_parameters['pool_kernel_size']
        iterCNNTestingModel.pool_stride = pool_stride = random_parameters['pool_stride']
        CNN_model = CNN(dropout_pr, 
                    num_hidden,
                    num_classes,
                    conv_pool_layers,
                    fc_layers,
                    conv_kernel_size,
                    conv_stride,
                    conv_padding,
                    pool_kernel_size,
                    pool_stride,
                    10,
                    activ_func,
                    l_func
    )

    # Since the CNN model can be capped off at a certain number of layers
    # we need to update the model to reflect this
    iterCNNTestingModel.conv_pool_layers = CNN_model.conv_pool_layers

    if val_print_explicit > 0:
        print("> CNN model parameters")
        print(CNN_model.parameters)
        # state_dict() maps each layer to its parameter tensor.
        print ("> CNN model's state dictionary")
        for param_tensor in CNN_model.state_dict():
            print(param_tensor, CNN_model.state_dict()[param_tensor].size())

    # To turn on/off CUDA if I don't want to use it.
    CUDA_enabled = True
    if (device.type == 'cuda' and CUDA_enabled):
        print("...Modeling CNN using GPU...")
        CNN_model = CNN_model.to(device=device) # sending to whaever device (for GPU acceleration)
        # CNN_model = CNN_model.to(device=device)
    else:
        print("...Modeling CNN using CPU...")
    
    ### Choose a gradient method
    # model hyperparameters and gradient methods
    # optim.SGD performs gradient descent and update the weigths through backpropagation.
    num_epochs = iterCNNTestingModel.epochs

    # Stochastic Gradient Descent (SGD) is used in this program.
    #+ You can choose other gradient methods (Adagrad, adadelta, Adam, etc.) and parameters

    if grad_method == "SGD":
        CNN_optimizer = optim.SGD(CNN_model.parameters(), lr=alpha, momentum=gamma)
    elif grad_method == "Adadelta":
        CNN_optimizer = optim.Adadelta(CNN_model.parameters(), lr=alpha, rho=rho)
    elif grad_method == "Adagrad":
        CNN_optimizer = optim.Adagrad(CNN_model.parameters(), lr=alpha)
    if val_print_explicit > 0:
        print("> CNN optimizer's state dictionary")
        for var_name in CNN_optimizer.state_dict():
            print(var_name, CNN_optimizer.state_dict()[var_name])
    
    
    ### Define a loss function: You can choose other loss functions

    if l_func == "crossEntropy":
        l_func = nn.CrossEntropyLoss()

    ### Train your networks
    if val_print_explicit > 0:
        print("............Training CNN................")
    is_MLP = False
    start_time = time()
    train_loss=train_ANN_model(num_epochs,
                               train_dataloader,
                               device,
                               CUDA_enabled,
                               is_MLP,
                               CNN_model,
                               l_func,
                               CNN_optimizer,
                               mini_batch_size,
                               num_train_batches,
                               val_print_explicit
    )
    training_time = time() - start_time
    if val_print_explicit > 0:
        print("............Testing CNN model................")

        print("> Input digits:")
        print(labels)
    predicted_digits, accuracy=test_ANN_model(device,
                                              CUDA_enabled, 
                                              is_MLP, 
                                              CNN_model, 
                                              test_dataloader,
                                              mini_batch_size,
                                              num_test_batches,
                                              val_print_explicit
    )
    if val_print_explicit > 0:
        print("> Predicted digits by CNN model")
        print(predicted_digits)

    iterCNNTestingModel.updateResults(training_time=training_time,accuracy=accuracy)
    testingResults["CNN"].append(iterCNNTestingModel)

if val_print_explicit > 0:
    print("All CNN tests completed:")

# with open(testingResultsFileName,"w",newline="") as file:
with open(cnntestingResultsFileName,"w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epochs",
                     "conv_pool_layers",
                     "fc_layers",
                     "conv_kernel_size",
                     "conv_stride",
                     "conv_padding",
                     "pool_kernel_size",
                     "pool_stride",
                     "hidden_neurons",
                     "mini_batch_size",
                     "activationFunction",
                     "loss_Function",
                     "grad_method",
                     "alpha_learning_rate",
                     "gamma_momentum",
                     "rho",
                     "dropout",
                     "training_time",
                     "accuracy"]
                     )
    if val_print_explicit > 0:
        for iterCNNresult in testingResults["CNN"]:
            print(iterCNNresult)

            writer.writerow(iterCNNresult.to_csv())

exit()