import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2
import torchvision.transforms as transforms
import os
import torch
from skimage.metrics import structural_similarity


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        #get images to use for SSIM calculation
        img_fake = convertTensorToImage(self.fake_B)
        img_real = convertTensorToImage(self.real_B)

        SSIM_score, diff = structural_similarity(img_fake, img_real, full = True, multichannel = True)

        #add SSIM loss to L1 loss
        self.loss_G_L1 += (1-SSIM_score) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def track_test_img(self, epoch):
        input_folder = "./datasets/EBSD/hexagonal/A/test/ipf_image_101/"
        arr_np = np.zeros((256,256,36), dtype = np.uint8)
        for i in range(1, 37):
            img_path = input_folder+"ipf_image_101_"+str(i)+".tif"
            img = cv2.imread(img_path, 0)
            img_arr = np.asarray(img)
            arr_np[:,:,i-1] = img_arr
        test_transform = get_test_transform()
        arr_tens = test_transform(arr_np)
        arr_tens_v2 = torch.zeros([1,36,256,256], dtype = torch.float32)
        arr_tens_v2[0,:,:,:] = arr_tens
        arr_tens_v2 -= 0.5
        arr_tens_v2 *= 2

        fake_tens = self.netG(arr_tens_v2)   #G(A)
        tens_prep = torch.zeros([36, 256, 256], dtype = torch.float32)
        tens_prep = fake_tens[0,:,:,:]

        img_transform = get_img_transform()
        fake_img = img_transform(tens_prep.cpu())

        directory = "./results/test_tracking/"

        if (not os.path.isdir(directory)):
            os.mkdir(directory)

        fake_img.save(directory+"test_image_101_epoch_"+str(epoch)+".tif")

def get_test_transform():
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    return transforms.Compose(transforms_list)

def get_img_transform():
    transforms_list = []
    transforms_list += [transforms.ToPILImage()]
    return transforms.Compose(transforms_list)

def convertTensorToImage(tens):
    #convert tensor to numpy array
    arr_np = np.zeros((256, 256, 3), dtype = np.float32)

    arr_np[:,:,0] = tens[0,0,:,:].cpu().detach().numpy()
    arr_np[:,:,1] = tens[0,1,:,:].cpu().detach().numpy()
    arr_np[:,:,2] = tens[0,2,:,:].cpu().detach().numpy()

    #format array properly
    arr_np += 1
    arr_np *= 127.5
    arr_np = arr_np.astype(np.uint8)

    #convert arrays to imgs for ssim calculation
    cv2.imwrite("./temp.tif", arr_np)
    img = cv2.imread("./temp.tif")

    return img
