from PIL import Image
import torch
import matplotlib.pyplot as plt 
from torchvision.transforms import transforms
import os

dir_target = os.path.join(os.path.expanduser('~'),'Datasets/df2k')
path_targets = [ os.path.join(dir_target, f) for f in os.listdir(dir_target) ] 


limit = 5 
size_target = 256 
plt.figure(figsize=(10,8))
for i, path_target in enumerate(path_targets):
    if limit  <= i:
        break;

    ax = plt.subplot(4, limit, i + 1)
    im = Image.open(path_target)
    im = transforms.CenterCrop(size_target)(im)
    ax.imshow(im)

    ax = plt.subplot(4, limit, limit + i + 1)
    im = Image.open(path_target)
    im = transforms.CenterCrop(size_target)(im)
    im = transforms.Resize(int(size_target / 4))(im)
    ax.imshow(im)

    ax = plt.subplot(4, limit, limit * 2 + i + 1)
    im = Image.open(path_target)
    im = transforms.CenterCrop(size_target)(im)
    im = transforms.Resize(int(size_target / 8))(im)
    ax.imshow(im)

    ax = plt.subplot(4, limit, limit * 3 + i + 1)
    im = Image.open(path_target)
    im = transforms.CenterCrop(size_target)(im)
    im = transforms.Resize(int(size_target / 16))(im)
    ax.imshow(im)


plt.tight_layout()
plt.show()

