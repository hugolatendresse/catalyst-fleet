"""
Model Type: CNN
Model Definition: PyTorch
Model Export: fx tracer
Model Ingestion: from_fx
Target: LLVM
Result: FAIL Downcast from relax.expr.Function to tir.PrimFunc failed.
"""

import sys
sys.path.append('/ssd1/htalendr/tvm/python')
# sys.path.append('/ssd1/htalendr/yolov5')
import tvm
from tvm import relax
import tvm.testing
from tvm.relax.frontend.torch import from_fx

import os
import torch
from random import randint
import torch.fx as fx
from torch.fx import wrap
from torch import fx # TODO importing fx twice but differently??
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.fx.proxy import Proxy
import matplotlib.pyplot as plt

data_path = 'shapes_data/'
model_file = 'shape_cnn/shape_classifier.pt'

# Import PyTorch libraries

# Other libraries we'll use
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

class PyTorchCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PyTorchCNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # self.drop = nn.Dropout2d(p=0.2) # TODO retrain without dropout?
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # # Ensure input is in the correct format (assumes already in NCHW if using PyTorch DataLoader)
        # if not isinstance(x, torch.Tensor):
        #     x = self.transformation(x).float() # Converts HWC -> CHW
        #     x = x.unsqueeze(0)  # Converts CHW -> NCHW
        #     x = Variable(x)

        # Forward pass through CNN layers
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x)) # used to be: x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        
        # Flatten the tensor before passing to the fully connected layer
        x = x.view(x.size(0), -1)  # Use x.size(0) to handle batch size dynamically
        x = self.fc(x)
        
        # Return log probabilities for classification
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

# Ensure input is in the correct format (NCHW torch tensor)
# Transformation assumes a single image input (not batched)
transformation = transforms.Compose([
    transforms.ToTensor(),  # Converts HWC -> CHW
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_torch = transformation(img_np).float() # Converts HWC -> CHW
img_torch = img_torch.unsqueeze(0)  # Converts CHW -> NCHW
img_torch = Variable(img_torch)

# Create a new model class and load the saved weights
model = PyTorchCNN()
model.load_state_dict(torch.load(model_file))

# Set the classifer model to evaluation mode
model.eval()

# Predict the class of the image
output = model(img_torch)
index = output.data.numpy().argmax()
print("According to pytorch inference, the image is a", classes[index])


### Conversion to IR

torch_model = PyTorchCNN()
torch_model.load_state_dict(torch.load(model_file))

input_info = [((1, 3, 128, 128), "float32")]


# Use FX tracer to trace the PyTorch model.
graph_module = fx.symbolic_trace(torch_model)


fx.symbolic_trace(torch_model).graph.print_tabular()
irmodule = from_fx(graph_module, input_info)
print(irmodule)
rt_lib_target = tvm.build(irmodule, target="llvm") # TODO why doesn't this work?
tvm_input = tvm.nd.array(img_np) # TODO should the input be img_np or img_torch or something else?
out = rt_lib_target["main"](tvm_input)
print(out)