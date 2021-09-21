#Import
import torch.nn as nn

#Define neural network
#Takes the depth, width, input size and output size as arguments
class Net(nn.Module):
    def __init__(self,depth,width,input_size,output_size):
        super(Net, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(input_size,width)]+[nn.Linear(width,width) for i in range(depth-1)]) #N hidden layers, N = depth-1: width -> width
        
        self.output = nn.Linear(width,output_size) #Output layer: width -> output_size

        self.act = nn.Tanh() 


    def forward(self, x):
        #Forward pass through all hidden layers with relu activation
        for i, layer in enumerate(self.layers):
          x = self.act(layer(x))
        
        x = self.output(x) #Output layer
        return x