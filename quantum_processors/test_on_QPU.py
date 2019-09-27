# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing a hybrid image classifier on IBM or Rigetti quantum processors."""

import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
# Pennylane
import pennylane as qml
from pennylane import numpy as np
# Other tools
import time
import os
import copy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Setting of the main parameters of the network model and of the training process. These should match the topology of the saved pre-trained model (quantum_weights_pt).
n_qubits = 4                            # number of qubits
step = 0.0004                           # learning rate
batch_size = 4                          # number of samples for each training step
num_epochs = 30                         # number of training epochs
q_depth = 6                             # depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1                # learning rate reduction applied every 10 epochs.                       
n_quantum_layers = 15                   # Keep 15 even if not all are used.
q_delta = 0.01                          # Initial spread of random quantum weights
rng_seed = 0                            # seed for random number generator
start_time = time.time()                # start of the computation timer
data_dir = '../data/hymenoptera_data'   # path of dataset


# Choose between the two quantum backends: 'ibm' or 'rigetti'.
# ========= QPU ==========
backend = 'ibm'          #
# backend = 'rigetti'    #
# ========================

# Set the chosen backend as a PennyLane device. 
if backend == 'ibm':
    token = '' # Insert your personal IBM token. Remove the token when sharing your code!
    dev = qml.device('qiskit.ibm', wires=n_qubits, backend='ibmqx4', ibmqx_token=token)

if backend == 'rigetti':
    dev_qpu = qml.device('forest.qpu', device='Aspen-4-4Q-A', shots=1024)

print('Device capabilities: ', dev.capabilities()['backend'])

# Configure PyTorch to use CUDA, only if available. Otherwise simply use the CPU.
print('Initializing backend device...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset loading
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),     # uncomment for data augmentation
        #transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                     data_transforms[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


# Initialize dataloader
torch.manual_seed(rng_seed)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                  batch_size=batch_size, shuffle=True) for x in ['train', 'val']}

# function to plot images
def imshow(inp, title=None):
    """Display image from tensor.

    Args:
        inp (tensor): input image.
        title (string): title of the image.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


# Hybrid transfer learning model (classical-to-quantum).
# 
# We first define some quantum layers that will compose the quantum circuit.

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates. 

    Args:
        nqubits (int): number of qubits.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
        
def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis. 

    Args:
        w (tensor): list of rotation angles. One for each qubit. 
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.

     Args:
        nqubits (int): number of qubits.
    """
    # In other words it should apply something like :
    #CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT  
    for i in range(0, nqubits - 1, 2): #loop over even indices: i=0,2,...N-2  
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2): #loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

# Let us define the quantum circuit by using the PennyLane `qnode` decorator . The structure is that of a typical variational quantum circuit:
# 1. All qubits are first initialized in a balanced superposition of *up* and *down* states, then they are rotated according to the input parameters (local embedding);
# 2. Successively a sequence of trainable rotation layers and constant entangling layers is applied. This block is responsible for the main computation necessary to solve the classification problem.
# 3. Eventually, for each qubit, the local expectation value of the Z operator is measured. This produces a classical output vector, suitable for additional post-processing.

@qml.qnode(dev, interface='torch')
def q_net(q_in, q_weights_flat):
        """Quantum cricuit

        Args:
            q_in (tensor): input features.
            q_weights_flat (tensor): variational parameters.
        Returns:
            tuple: expectation values of PauliZ for each qubit.
        """
        # reshape weights
        q_weights = q_weights_flat.reshape(n_quantum_layers, n_qubits)
        H_layer(n_qubits)   # Start from state |+> , unbiased w.r.t. |0> and |1>
        RY_layer(q_in)      # Embed features in the quantum node
       
        # sequence of trainable variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k + 1])

        # expectation values in the Z basis
        exp_vals = [qml.expval.PauliZ(position) for position in range(n_qubits)]
        return tuple(exp_vals)

# We can now define a custom `torch.nn.Module` representing a *dressed* quantum circuit.
# This is is a concatenation of:
# 1. A classical pre-processing layer (`nn.Linear`)
# 2. A classical activation function (`F.tanh`)
# 3. A constant `np.pi/2.0` scaling factor.
# 2. The previously defined quantum circuit (`q_net`)
# 2. A classical post-processing layer (`nn.Linear`)
# 
# The input of the module is a batch of vectors with 512 real parameters (features) and the output is a batch of vectors with two real outputs (associated with the two classes of images: *ants* and *bees*).

class Quantumnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_net = nn.Linear(512, n_qubits)
            self.q_params = nn.Parameter(q_delta * torch.randn(n_quantum_layers * n_qubits))
            self.post_net = nn.Linear(n_qubits, 2)

        def forward(self, input_features):
            """Full classical-quantum network

            Args:
                self.
                input_features (tensor): input image.
            Returns:
                tuple: output logits of the hybrid network.
            """
            pre_out = self.pre_net(input_features) 
            q_in = F.tanh(pre_out) * np.pi / 2.0   
            # apply the quantum circuit to each element of the batch, and append to q_out
            q_out = torch.Tensor(0, n_qubits)
            q_out = q_out.to(device)
            for elem in q_in:
                q_out_elem = q_net(elem,self.q_params).float().unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))
            return self.post_net(q_out)

# We are finally ready to build our full hybrid classical-quantum network. We follow the *transfer learning* approach:
# First load the classical pre-trained network *ResNet18* from the `torchvision.models` zoo. The model is downloaded from Internet and it may take a long time (only the first time). 
model_hybrid = torchvision.models.resnet18(pretrained=True)
# Freeze all the weights since they should not be trained.
for param in model_hybrid.parameters():
    param.requires_grad = False
# Replace the last fully connected layer with our trainable dressed quantum circuit (`Quantumnet`). 
model_hybrid.fc = Quantumnet()


# use CUDA or CPU according to the "device" object.
model_hybrid = model_hybrid.to(device)

# Load model from file
model_hybrid.fc.load_state_dict(torch.load("quantum_weights.pt", map_location='cpu'))

# We apply the model to the test dataset to compute the associated loss and accuracy.
criterion = nn.CrossEntropyLoss()
running_loss = 0.0
running_corrects = 0
n_batches = dataset_sizes['val'] // batch_size
it = 0

print('Results of the model testing on a real quantum processor.',  file=open('results_' + backend + '.txt', 'w'))
print('QPU backend: ' + backend,  file=open('results_' + backend + '.txt', 'a'))

for inputs, labels in dataloaders['val']:
                    model_hybrid.eval()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    batch_size_ = len(inputs)
                    with torch.set_grad_enabled(False):
                        outputs = model_hybrid(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == labels.data).item()
                    running_corrects += batch_corrects
                    print('Iter: {}/{}'.format(it + 1, n_batches + 1), end='\r', flush=True)
                    # log to file
                    print('Iter: {}/{}'.format(it + 1, n_batches + 1), end='\r', flush=True, file=open('results_' + backend + '.txt', 'a'))
                    it+=1

epoch_loss = running_loss / dataset_sizes['val']
epoch_acc = running_corrects / dataset_sizes['val']
print('\nTest Loss: {:.4f} Test Acc: {:.4f}        '.format(epoch_loss, epoch_acc))
# log to file
print('\nTest Loss: {:.4f} Test Acc: {:.4f}        '.format(epoch_loss, epoch_acc), file=open('results_' + backend + '.txt', 'a'))

# Compute and the visualize the predictions for a batch of test data.
# The figure is saved as a .png file in the working directory.
images_so_far = 0
num_images = batch_size
fig = plt.figure('Predictions')
model_hybrid.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_hybrid(inputs)
        _, preds = torch.max(outputs, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('[{}]'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])
            if images_so_far == num_images:
                fig.savefig('predictions_' + backend + '.png')
                break
        if images_so_far == num_images:
            break
