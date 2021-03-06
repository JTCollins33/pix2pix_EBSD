import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, mod_get_transform
from data.image_folder import make_dataset, mod_make_dataset
from PIL import Image
import cv2
import numpy as np
import torch

nImages = 36

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    #THIS FUNCTION HAS BEEN MODIFIED
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #MANUALLY PUT IMAGE DIRECTORIES, CHANGE FOR LATER
        if opt.phase == "test":
            self.dir_A = opt.dataroot + "/A/test"
            self.dir_B = opt.dataroot + "/B/test"
        else:
            self.dir_A = opt.dataroot + "/A"
            self.dir_B = opt.dataroot + "/B"
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.A_paths = sorted(mod_make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    #THIS FUNCTION HAS BEEN MODIFIED
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        #read image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        found = False
        index_string = ''
        i = -1
        while (not found):
          if (A_path[i] == '_'):
            found = True
          else:
            index_string = A_path[i]+index_string
          i = i-1
        real_index = int(index_string)


        B2 = cv2.imread(B_path)
        imgB2 = np.asarray(B2)
        B = np.true_divide(imgB2, 255.0).astype('float32')
        # B = Image.fromarray(imgB2)
        # B = Image.fromarray(extendZerosRGB(imgB2))
        #do for first image to get base tensor to concatenate to
        first_A_Path = A_path+"/ipf_image_"+str(real_index)+"_1.tif"
        A2 = cv2.imread(first_A_Path, 0)
        imgA2 = np.asarray(A2)
        mergeA = np.true_divide(imgA2, 255.0).astype('float32')
        
        A_transform = mod_get_transform(self.opt, grayscale = (self.input_nc != 3))
        B_transform = mod_get_transform(self.opt, grayscale = (self.output_nc == 1))
        B = B_transform(B)
        mergeA = A_transform(mergeA)

        #do for rest of images
        for i in range(2, nImages+1):
            current_A_Image_Path = A_path + "/ipf_image_"+str(real_index)+"_"+str(i)+".tif"
            A2 = cv2.imread(current_A_Image_Path, 0)
            imgA2 = np.asarray(A2)
            A = np.true_divide(imgA2, 255.0).astype('float32')
            A = A_transform(A)
            mergeA = torch.cat((mergeA, A))
        return {'A': mergeA, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)
