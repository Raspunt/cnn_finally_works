import torchvision.datasets as dset
import os

class ImageFolderEX(dset.ImageFolder):
    def __getitem__(self, index):


        path, target = self.samples[index]
        # print(path)

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return sample, target

    def __len__(self) -> int:

        if self.samples is None:
            self.samples = []

        return len(self.samples)
       