import torch
import torchvision
import utils
import torchvision.transforms as transforms
from models.enet import ENet
from PIL import Image

num_classes = 12
device = 'cuda'
model_path = 'save/ENet_CamVid/ENet'
w = 480
h = 264
dataset_dir = '/tmp/dataset/CamVid'
batch_size = 8
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

# Create a batch
for i, batch in enumerate(val_loader):
    # Get the inputs and labels
    inputs = batch[0].to(device)

    break

# Intialize ENet
model = ENet(num_classes).to(device)

# Load the stored model parameters to the model instance
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

traced_net = torch.jit.trace(model, inputs)
