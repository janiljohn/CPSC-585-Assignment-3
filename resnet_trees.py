import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
from pillow_heif import register_heif_opener

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# contents = os.listdir('Tree Data')

# register_heif_opener()

# for i, item in enumerate(contents):
#   print(i, item)
#   img = Image.open('Tree Data/'+item)
#   img.convert('RGB').save(f'{os.getcwd()}/Images/{i+1}.jpg')

contents = os.listdir('Images')

for item in contents:
  img = Image.open(item)
  

print('PyTorch version:', torch.__version__)