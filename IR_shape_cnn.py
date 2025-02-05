import sys
sys.path.append('/ssd1/htalendr/tvm/python')
sys.path.append('/ssd1/htalendr/yolov5')
from tvm import relax

import torch.fx as fx
from torch.fx import wrap


import numpy as np
import torch
from torch import fx
import tvm
import tvm.testing
from tvm.relax.frontend.torch import from_fx

from torch.fx.proxy import Proxy
import matplotlib.pyplot as plt
import os
from random import randint



import torch

from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

import numpy as np
import requests
from PIL import Image

data_path = 'shapes_data/'
model_file = 'shape_cnn/shape_classifier.pt'

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# Other libraries we'll use
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PyTorchCNN(nn.Module):
    # Constructor
    def __init__(self, num_classes=3):
        super(PyTorchCNN, self).__init__()
        
        # Our images are RGB, so input channels = 3. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # So our feature tensors are now 32 x 32, and we've generated 24 of them
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to  the probability for each class
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

        self.transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def forward(self, x):
        # This forward pass assume a numpy input (not a image or tensor)

        if not isinstance(x, Proxy):  # Skip transformation during FX tracing
            x = self.transformation(x).float().unsqueeze(0)
            x = Variable(x)


        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
      
        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        
        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        
        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return log_softmax tensor 
        return F.log_softmax(x, dim=1)
    

# Get the class names
classes = os.listdir(data_path)
classes.sort()


# Function to create a random image (of a square, circle, or triangle)
def create_image (size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw
    
    xy1 = randint(10,40)
    xy2 = randint(60,100)
    col = (randint(0,200), randint(0,200), randint(0,200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if shape == 'circle':
        draw.ellipse([(xy1,xy1), (xy2,xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    else: # square
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    del draw
    
    return np.array(img)

# Create a random test image
classnames = os.listdir(data_path)
classnames.sort()
shape = classnames[randint(0, len(classnames)-1)]
img_np = create_image((128,128), shape)

# Display the image
plt.axis('off')
plt.imshow(img_np)

# Create a new model class and load the saved weights
model = PyTorchCNN()
model.load_state_dict(torch.load(model_file))

# Set the classifer model to evaluation mode
model.eval()

# Predict the class of the image
output = model(img_np)
index = output.data.numpy().argmax()
print("According to pytorch inference, the image is a", classes[index])


### Conversion to IR

torch_model = PyTorchCNN()
torch_model.load_state_dict(torch.load(model_file))

input_info = [((128, 128), "float32")]


# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)


fx.symbolic_trace(torch_model).graph.print_tabular()


irmodule = from_fx(graph_module, input_info)
print(irmodule)

rt_lib_target = tvm.build(irmodule, target="llvm") # TODO why doesn't this work?
tvm_input = tvm.nd.array(img_np)
out = rt_lib_target["main"](tvm_input)
print(out)