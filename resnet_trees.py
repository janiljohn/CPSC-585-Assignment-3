import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
# from pillow_heif import register_heif_opener

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])



# # contents = os.listdir('Tree Data')

# # register_heif_opener()

# # for i, item in enumerate(contents):
# #   print(i, item)
# #   img = Image.open('Tree Data/'+item)
# #   img.convert('RGB').save(f'{os.getcwd()}/Images/{i+1}.jpg')

# contents = os.listdir('Images')

# for item in contents:
#   img = Image.open(item)
  

# print('PyTorch version:', torch.__version__)


model = models.resnet50(pretrained=True)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# with open("tree_data_labels.txt") as f:
#     labels = [line.strip() for line in f.readlines()]

dir = "Tree Data JPG"

for filename in os.listdir(dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):

        path = os.path.join(dir, filename)
        image = Image.open(path)
        img_preprocessed = preprocess(image)
        input_batch = img_preprocessed.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        probability = torch.nn.functional.softmax(output[0], dim=0)
        
        predicted_class_index = torch.argmax(probability).item()

        # predict_label = labels[predicted_class_index%17]
        predict_label = predicted_class_index

        print("Image:", filename)
        print("Predicted label:", predict_label)