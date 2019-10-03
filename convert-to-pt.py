import torch
import torchvision
import utils
import torchvision.transforms as transforms
import transforms as ext_transforms
from models.enet import ENet
from PIL import Image
import torch.utils.data as data
from data import CamVid as dataset

num_classes = 12
device = 'cuda'
model_path = 'save/ENet_CamVid/ENet'
w = 480
h = 264
dataset_dir = '/tmp/dataset/CamVid'
batch_size = 1
workers = 4

# Image transform
image_transform = transforms.Compose(
    [transforms.Resize((h, w)),
    transforms.ToTensor()])

label_transform = transforms.Compose([
    transforms.Resize((h, w), Image.NEAREST),
    ext_transforms.PILToLongTensor()
])

# Dataset
val_set = dataset(
    dataset_dir,
    mode='val',
    transform=image_transform,
    label_transform=label_transform)
val_loader = data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers)

# Intialize ENet
model = ENet(num_classes).to(device)
print("ENet initialized")

# Create a batch
for i, batch in enumerate(val_loader):
    # Get the inputs and labels
    inputs = batch[0].to(device)

    break


# Load the stored model parameters to the model instance
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# Trace the network
inputs = torch.rand(1, 3, h, w).to(device)
print(inputs)
traced_net = torch.jit.trace(model, inputs)
# traced_net = torch.jit.trace(model, )
print("Trace done")

# Save the module
traced_net.save("ENet_h{}_w{}.pt".format(h,w))
print("ENet_h{}_w{}.pt is exported".format(h,w))
