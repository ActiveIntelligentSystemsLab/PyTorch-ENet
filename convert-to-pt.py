import torch
from models.enet import ENet

num_classes = 12
device = 'cuda'
model_path = 'save/ENet_CamVid/ENet'
w = 480
h = 264
batch_size = 1

# Intialize ENet
model = ENet(num_classes).to(device)
print("ENet initialized")

# Load the stored model parameters to the model instance
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# Trace the network with random data
inputs = torch.rand(1, 3, h, w).to(device)
print(inputs)
traced_net = torch.jit.trace(model, inputs)
print("Trace done")

# Save the module
traced_net.save("ENet_h{}_w{}.pt".format(h,w))
print("ENet_h{}_w{}.pt is exported".format(h,w))
