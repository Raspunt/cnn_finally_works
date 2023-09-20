import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


import PIL
import matplotlib.pyplot as plt

# Assuming you have defined the ImageFolderEX and Net1 classes in their respective files
from ImageFolderFix import ImageFolderEX
from cnn_model import Net

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DogsAndCats():
    IMG_SIZE = (50, 50)
    classes = ("cat","dog")
    batch = 32

    dataset_path = './datasets/catAndDogs.pth'

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.net = Net()  # Initialize the neural network model


    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform


    def make_training_data(self):
        transform = self.get_transform()

        self.dataset = ImageFolderEX(self.data_dir, transform=transform)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.dataset, batch_size=self.batch, shuffle=True, num_workers=4)

        print("Data preparation is done")

    def train_the_nn(self):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(),
                               lr=0.001,
                               weight_decay=0.0)
        size = len(self.dataset)

        for epoch in range(2):  # Loop over the dataset multiple times
            print(f"Epoch {epoch + 1}/{2}")
            for i, (inputs, labels) in enumerate(self.train_dataloader):

                optimizer.zero_grad()
                pred = self.net(inputs)
                loss = loss_fn(pred, labels)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    loss, current = loss.item(), i * len(inputs)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                        
                        
        torch.save(self.net.state_dict(), self.dataset_path)

        print('Finished Training')


    def get_prediction(self,path_img : str):
        img = PIL.Image.open(path_img)
        transform =  self.get_transform()
        img = transform(img).unsqueeze(0)

        self.net.load_state_dict(torch.load(self.dataset_path))
        self.net.eval()

        with torch.no_grad():
            output = self.net(img)  
            _, predicted_class = output.max(1) 
            print(predicted_class)

    

if __name__ == "__main__":
    data_dir = "./raw_datasets/PetImages/"
    dac = DogsAndCats(data_dir)
    # dac.make_training_data()
    # dac.train_the_nn()

    dac.get_prediction("./dog.jpg")

