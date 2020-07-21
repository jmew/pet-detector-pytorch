import os
import random
import re
import shutil
import tarfile
from os.path import basename, isfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ipywidgets import interact
from PIL import Image
from torchvision import datasets, models, transforms


# Fetch a file from uri, unzip and untar it into its own directory.
def fetch_and_untar(uri, path):
    # If the data set exists, skip this step
    if os.path.isdir(path):
        return

    # Parse the uri to extract the local filename
    parsed_uri = urlparse(uri)
    local_filename = basename(parsed_uri.path)

    # If file is not already on disk, retrieve from uri
    if not isfile(local_filename):
        with urlopen(uri) as response:
            with open(local_filename, 'bw+') as f:
                shutil.copyfileobj(response, f)

    # Expand the archive
    with tarfile.open(local_filename) as tar:
        tar.extractall()
    
    move_images_into_labelled_directories(path)

def move_images_into_labelled_directories(image_dir):
    images_path = Path(image_dir)
    extract_breed_from_filename = re.compile(r'([^/]+)_\d+.jpg$')

    for filename in os.listdir(image_dir):
        match = extract_breed_from_filename.match(filename)
        if match is not None:
            breed = match.group(1)
            if not os.path.exists(images_path / breed):
                os.makedirs(images_path / breed)
            src_path = images_path / filename
            dest_path = images_path / breed / filename
            shutil.move(src_path, dest_path)

# Split into training and validation folders (80/20 split)
def split_train_val(path):
    if os.path.isdir(path):
        return
        
    for subdir, dirs, files in os.walk(path):
        if subdir == path:
            if not os.path.exists(os.path.join(path, 'val')):
                os.mkdir(os.path.join(path, 'val'))

            if not os.path.exists(os.path.join(path, 'train')):
                os.mkdir(os.path.join(path, 'train'))
            continue
        for x in range(40):
            fil = random.choice(os.listdir(subdir))
            os.replace(os.path.join(subdir, fil), os.path.join(path, os.path.join('val', fil)))

        new_dir = os.path.join(os.path.split(subdir)[0], os.path.join('train', os.path.split(subdir)[1]))
        shutil.move(subdir, new_dir)

    move_images_into_labelled_directories(os.path.join(os.getcwd(), os.path.join('images', 'val')))

def get_sample_images_for_each_species(dirname):
    d = Path(os.path.join(dirname, 'train'))
    species_dirs = [d for d in d.iterdir() if d.is_dir()]
    species_images_and_labels = []
    for species_dir in species_dirs:
        for image_path in species_dir.iterdir():
            image = Image.open(image_path)
            image_label = species_dir.parts[-1].lower().replace('_', ' ')
            species_images_and_labels.append((image, image_label))
            break
    return species_images_and_labels

def plot_images_in_grid(images_data, number_columns):
    f, subplots = plt.subplots(len(images_data) // number_columns + 1, number_columns)
    f.set_size_inches(16, 16)

    row = 0
    col = 0

    for record in images_data:
        subplot = subplots[row, col]
        subplot.imshow(record[0])
        subplot.set_axis_off()
        subplot.set_title(record[1], color='#358CD6')
        col += 1
        if col == number_columns:
            row += 1
            col = 0

    for c in range(col, number_columns):
        subplots[row, c].set_axis_off()

def browse_images(digits):
    n = len(digits)
    def view_image(i):
        plt.imshow(digits[i][0], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %s' % digits[i][1])
        plt.show()
    interact(view_image, i=(0,n-1))

def transform_images_to_tensors():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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

    data_dir = 'images/'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    save_labels(class_names)

    return dataloaders, dataset_sizes, class_names

def save_labels(class_names):
    with open("labels.txt", "w") as output:
        for row in class_names:
            output.write(str(row).replace('_', ' ').title() + '\n')

def setup_model(model_ft, device):
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 37)
    model_ft = model_ft.to(device)

def save_best_model(model_ft):
    if not os.path.exists(os.path.join(os.getcwd(), 'model')):
        os.mkdir(os.path.join(os.getcwd(), 'model'))

    save_dir = os.path.join(os.getcwd(), os.path.join('model', 'checkpoint.pth'))

    torch.save(model_ft, save_dir)

def use_gpu_if_avail():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
