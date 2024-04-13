import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


###################### Designing an ANN architectures #########################

class MLP(nn.Module): # All models should inherit from nn.Module
    # This part can be changed based on the design decision.
    def __init__(self, 
                 num_input, 
                 layers, 
                 hidden_size, 
                 num_classes,
                 dropout_pr=0.05,
                 af=AF.relu,
                 loss_Function="crossEntropy"
        ): # Define our ANN structures here
        self.af = af
        self.loss_Function = loss_Function
        super(MLP, self).__init__()
        # nn.Linear(in_features, out_features, bias): y = w^Tx + bias
        # using ModuleList to store a list of layers in a model
        self.hiddens = nn.ModuleList([nn.Linear(num_input, hidden_size)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_pr)])
        for i in range(layers - 1):
            self.hiddens.append(nn.Linear(hidden_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout_pr))
        self.output = nn.Linear(hidden_size, num_classes)
        
        # The model structure can be also defined using "sequential" function
        # self.seq_linear=nn.Sequential(nn.Linear(num_input, hidden1_size),nn.RELU(),nn.Linear(hidden1_size, num_classes))

    # Define "forward" function to perform the computation for input x and return output(s).
    # The function name "forward" is required by Pytorch.
    def forward(self, x):
        # In this implementation, the activation function is reLU, but you can try other functions
        # torch.nn.functional modeule consists of all the activation functions and output functions
        for hidden, dropout in zip(self.hiddens, self.dropouts):
            x = self.af(hidden(x))
            x = dropout(x)

        if self.loss_Function != "crossEntropy":
            max_x = AF.log_softmax(x)
            x = self.output(max_x)
        else:
            x = self.output(x)
        # AF.softmax() is NOT needed when CrossEntropyLoss() is used as it already combines both LogSoftMax() and NLLLoss()
        # return self.seq_linear(x) # If the model structrue is define by sequential function.
        return x

### CNN architecture
### CNN architecture
class CNN(nn.Module):
    # The probability of dropout, number of hidden nodes, number of output classes
    def __init__(
            self, 
            dropout_pr:float, 
            num_hidden:int,
            num_classes:int,
            conv_pool_layers:int, 
            fc_layers:int, 
            conv_kernel_size:int = 5, 
            conv_stride:int = 1,
            conv_padding:int = 0,
            pool_kernel_size:int = 4, 
            pooling_stride:int=4,
            conv_channels:int=10,
            af=AF.relu,
            loss_Function="crossEntropy"
        ):
        super(CNN, self).__init__()
        # # Conv2d is for two dimensional convolution (which means input image is 2D)
        # # Conv2d(in_channels=, out_channels=, kernel_size=, stride=1, padding=0)
        # # in_channels=1 if grayscale, 3 for color; out_channels is # of output channels (or # of kernels)
        # # Ouput size for W from CONV = (W-F+2P)/S+1 where W=input_size, F=filter_size, S=stride, P=padding
        # # max_pool2d(input_tensor, kernel_size, stride=None, padding=0) for 2D max pooling
        # # Output size for W from POOL = (W-F)/S+1 where S=stride=dimension of pool
        # # K is # of channels for convolution layer; D is # of channels for pooling layer
        # Precalculate W's
        self.fail = False 
        self.af = af
        self.loss_Function = loss_Function
        self.conv_pool_layers = conv_pool_layers
        self.W_conv = int(((28 - conv_kernel_size + (2 * (conv_padding))) / conv_stride) + 1)
        print(f"W_conv: {self.W_conv}")
        self.W_pool = int(((self.W_conv - pool_kernel_size) / pooling_stride) + 1)
        print(f"W_pool: {self.W_pool}")
        if self.W_pool <= 1 or self.W_conv <= 1:
            print("Error: Initial convolutional and pooling layers are too deep for the image size. Exiting...")
            self.fail = True
            return
        self.conv = nn.ModuleList([nn.Conv2d(1, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)]) # K=D=10, output_size W=(28-5)/1+1=24 (24x24), (default Stride=1)
        self.pool = nn.ModuleList([nn.MaxPool2d(pool_kernel_size, pooling_stride)]) # W = (24-4)/4+1=6 (6x6), S=4 (pool dimension) since no overlapping regions
        self.dropout_conv = nn.ModuleList([nn.Dropout2d(dropout_pr)]) # to avoid overfitting by dropping some nodes
        for i in range(conv_pool_layers - 1):
            W_conv_temp = int(((self.W_pool - conv_kernel_size + (2 * (conv_padding))) / conv_stride) + 1)
            W_pool_temp = int(((W_conv_temp - pool_kernel_size) / pooling_stride) + 1)
            if W_pool_temp <= conv_kernel_size or W_conv_temp <= pool_kernel_size:
                print(f"Warning: Convolutional and pooling layers are too deep for the image size. Capping at layer {i+1}.")
                self.conv_pool_layers = i+1
                break
            self.W_conv = W_conv_temp
            self.W_pool = W_pool_temp
            print(f"W_conv: {self.W_conv}")
            print(f"W_pool: {self.W_pool}")
            self.conv.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding))
            self.pool.append(nn.MaxPool2d(pool_kernel_size, pooling_stride))
            self.dropout_conv.append(nn.Dropout2d(dropout_pr))
        #+ You can add more convolutional and pooling layers
        # Fully connected layer after convolutional and pooling layers
        self.num_flatten_nodes = conv_channels*(self.W_pool*self.W_pool) # Flatten nodes from 10 channels and 6*6 pool_size = 10*6*6=360
        self.fc = nn.ModuleList([nn.Linear(self.num_flatten_nodes, num_hidden)])
        self.dropout_fc = nn.ModuleList([nn.Dropout2d(dropout_pr)])
        for _ in range(fc_layers - 1):
            self.fc.append(nn.Linear(num_hidden, num_hidden))
            self.dropout_fc.append(nn.Dropout(dropout_pr))
        #+ You can add more hidden layers here if necessary



        self.out = nn.Linear(num_hidden, num_classes) # the output nodes are 10 classes (10 digits)
        
    def forward(self, x):
        for conv, pool, dropout in zip(self.conv, self.pool, self.dropout_conv):
            x = self.af(pool(conv(x)))
            x = dropout(x)
        x = x.view(-1, self.num_flatten_nodes)
        for fc, dropout in zip(self.fc, self.dropout_fc):
            x = self.af(fc(x))
            x = dropout(x)
        if self.loss_Function != "crossEntropy":
            max_x = AF.log_softmax(x)
            x = self.out(max_x)
        else:
            x = self.out(x)
        return x
    
# To display some images
def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    #print("Matrix for one image:")
    #print(images[1][0])
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
    plt.show()

# Training function


def train_ANN_model(num_epochs, 
                    training_data, 
                    device, 
                    CUDA_enabled, 
                    is_MLP, 
                    ANN_model, 
                    loss_func, 
                    optimizer,
                    mini_batch_size,
                    num_train_batches,
                    val_print_explicit=0
    ):
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
def test_ANN_model(device,
                   CUDA_enabled,
                   is_MLP,
                   ANN_model,
                   testing_data,
                   mini_batch_size,
                   num_test_batches,
                   val_print_explicit=0
    ):
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