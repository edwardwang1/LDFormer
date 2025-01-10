from torch.utils.data import Dataset
import os
import numpy as np
import torch

class Volumes(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.dir_files = os.listdir(self.directory)
        self.transform = transform

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.dir_files)
        

    def __getitem__(self, idx):
        # Load the image from the file
        # Filename based on the index
        volumes = torch.from_numpy(np.load(os.path.join(self.directory, self.dir_files[idx])))
        if self.transform:
            volumes = self.transform(volumes)
        #return volumes.copy()
        return volumes

class Context(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.dir_files = os.listdir(self.directory)

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.dir_files)

    def __getitem__(self, idx):
        # Load the image from the file
        # Filename based on the index
        context = np.loadtxt(os.path.join(self.directory, self.dir_files[idx]), delimiter=",").astype(int)
        #context = context.reshape((1, context.shape[0]))
        return context
    
class ContextFromDirectory(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.dir_files = os.listdir(self.directory) 

        self.contexts = []
        for file in self.dir_files:
            context = np.loadtxt(os.path.join(self.directory, file), delimiter=",").astype(int)
            self.contexts.append(context)

        self.contexts = np.vstack(self.contexts)
        #np.random.shuffle(self.contexts)

    def __len__(self):  # The length of the dataset is important for iterating through it
        return self.contexts.shape[0]
    
    def __getitem__(self, idx):
        # Load the image from the file
        # Filename based on the index
        context = self.contexts[idx]
        #context = context.reshape((1, context.shape[0]))
        return context
            

class ContextForDiffusion(Dataset):
    def __init__(self, dose_directory, oar_directory, ide_directory, shape, n_codes_dose, n_codes_oar):
        self.dose_directory = dose_directory
        self.dose_dir_files = os.listdir(self.dose_directory)
        self.oar_directory = oar_directory
        self.oar_dir_files = os.listdir(self.oar_directory)
        self.ide_directory = ide_directory
        self.ide_dir_files = os.listdir(self.ide_directory)

        self.shape = shape
        self.n_codes_dose = n_codes_dose
        self.n_codes_oar = n_codes_oar

    def __len__(self):  # The length of the dataset is important for iterating through it
        return len(self.dose_dir_files)

    def __getitem__(self, idx):
        # Load the image from the file
        # Filename based on the index
        dose_context = np.loadtxt(os.path.join(self.dose_directory, self.dose_dir_files[idx]), delimiter=",").astype(int)
        dose_context = torch.from_numpy(dose_context).view(self.shape)
        dose_context = dose_context - 1
        dose_context = (2 * (dose_context) / (self.n_codes_dose - 1)) - 1

        oar_context = np.loadtxt(os.path.join(self.oar_directory, self.oar_dir_files[idx]), delimiter=",").astype(int)
        oar_context = torch.from_numpy(oar_context).view(self.shape)
        oar_context = oar_context - 1
        oar_context = (2 * (oar_context) / (self.n_codes_oar - 1)) - 1

        ide_context = np.loadtxt(os.path.join(self.ide_directory, self.ide_dir_files[idx]), delimiter=",").astype(int)
        ide_context = torch.from_numpy(ide_context).view(self.shape)
        ide_context = ide_context - 1
        ide_context = (2 * (ide_context) / (self.n_codes_dose - 1)) - 1


        return np.stack([dose_context, oar_context, ide_context], axis=0)



if __name__ == "__main__":
    pass