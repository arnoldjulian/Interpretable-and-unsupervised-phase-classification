import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
import utils
import conf


# get magnitude of the discrete Fourier transform of input configuration
def fouriertrafo(image):
    dim = image.shape[0]
    ft_image = np.fft.fft2(image)

    ft_image_new = np.zeros((dim, dim), dtype=complex)
    ft_image_new[0:int(dim/2), 0:int(dim/2)] = ft_image[int(dim/2):dim, int(dim/2):dim]
    ft_image_new[int(dim/2):dim, 0:int(dim/2)] = ft_image[0:int(dim/2), int(dim/2):dim]
    ft_image_new[0:int(dim/2), int(dim/2):dim] = ft_image[int(dim/2):dim, 0:int(dim/2)]
    ft_image_new[int(dim/2):dim, int(dim/2):dim] = ft_image[0:int(dim/2), 0:int(dim/2)]
    image_ft_magn = np.abs(ft_image_new)

    return image_ft_magn

# characterizes a dataset for PyTorch, see also https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class DatasetFKM(data.Dataset):

    def __init__(self, files, transform=None, mean=None, std=None, input_type='ft'):

        # files that should be loaded in an epoch
        self.files = files

        # symmetry transformation
        self.transform = transform

        # standardization
        self.mean = mean

        self.std = std

        # input type
        self.input_type = input_type

    def __len__(self):

        # denotes the total number of samples
        return len(self.files)

    def __getitem__(self, index):
        # generates one sample (index) of data

        # select sample
        file = self.files[index]

        # load it
        content = np.loadtxt(file)

        if self.input_type == 'corr':
            input_np = np.loadtxt(
                str(file)[:-8] + 'corr_func.dat'.format(self.input_type)).reshape(-1)
            X = torch.from_numpy(input_np).view(1, -1).float()

        # when considering the raw configuration or the magnitude of the discrete FT as input, apply a random transformation of p4m to the configuration beforehand
        elif self.input_type == 'ft' or self.input_type == 'raw':
            if self.transform:
                X = torch.from_numpy(content).view(1, conf.dim, conf.dim).float()

                X = self.transform(X)

                content = X.cpu().numpy().reshape(content.shape)

            if self.input_type == 'ft':
                content = fouriertrafo(content.reshape(conf.dim, conf.dim)).reshape(conf.dim**2)

                X = torch.from_numpy(content).view(1, conf.dim, conf.dim).float()

            else:
                X = torch.from_numpy(content).view(1, conf.dim, conf.dim).float()

        # obtain the underlying system parameters p0
        label = (file.split('/')[-4:-2])
        U = float(label[0][1:])
        nf = float(label[1][2:])

        if conf.dim_parameter_space == 1:
            p0 = torch.tensor([U*conf.U_scale]).float()
        elif conf.dim_parameter_space == 2:
            p0 = torch.tensor([U*conf.U_scale, nf*conf.nf_scale]).float()
        else:
            print('Parameter space dimension is not supported!')

        # return the standardized input and the corresponding true underlying parameter values
        return torch.div(torch.sub(X, self.mean), self.std), p0

# provisional functions for translating configurations


def prov_roll_1(x):
    return utils.roll(x, 1, np.random.randint(conf.dim))


def prov_roll_2(x):
    return utils.roll(x, 2, np.random.randint(conf.dim))

# provisional functions for rotating configurations


def prov_rotate(x):
    return utils.rotate(x, np.random.randint(4))

# provisional functions for reflecting configurations


def prov_mirror_1(x):
    return utils.mirror(x, 1, np.random.randint(2))


def prov_mirror_2(x):
    return utils.mirror(x, 2, np.random.randint(2))

# define class for easy usage of data transformations, where a transformation for a particular type of input is constructed by composing the appropriate random symmetry transformations


class Transformations():

    def __init__(self):

        if conf.input_type == 'ft':

            self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Lambda(prov_rotate),
                    transforms.Lambda(prov_mirror_1),
                    transforms.Lambda(prov_mirror_2)
                ]),
                'test': transforms.Compose([
                    transforms.Lambda(prov_rotate),
                    transforms.Lambda(prov_mirror_1),
                    transforms.Lambda(prov_mirror_2)
                ])
            }

        else:
            self.data_transforms = {
                'train': transforms.Compose([
                    transforms.Lambda(prov_roll_1),
                    transforms.Lambda(prov_roll_2),
                    transforms.Lambda(prov_rotate),
                    transforms.Lambda(prov_mirror_1),
                    transforms.Lambda(prov_mirror_2)
                ]),
                'test': transforms.Compose([
                    transforms.Lambda(prov_roll_1),
                    transforms.Lambda(prov_roll_2),
                    transforms.Lambda(prov_rotate),
                    transforms.Lambda(prov_mirror_1),
                    transforms.Lambda(prov_mirror_2)
                ])
            }

    def get_dict(self, phase):
        return self.data_transforms[phase]
