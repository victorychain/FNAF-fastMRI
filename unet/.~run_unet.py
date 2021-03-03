"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.args import Args
from common.subsample import create_mask_for_mask_type
from common.utils import save_reconstructions
from data import transforms
from data.mri_data import SliceData
from unet_model import UnetModel

def make_IRIM():
    from irim import IRIM
    from irim import InvertibleUnet
    from irim import MemoryFreeInvertibleModule



    def grad_fun(x_est,y):
        """
        Defines the gradient function for a denoising problem with White Noise.
        This function demonstrates the use of  Pytorch's autograd to calculate the gradient.
        In this example, the function is equivalent to
        def grad_fun(x_est,y):
            return x_est - y
        :param x_est: Tensor, model estimate of x
        :param y: Tensor, noisy measurements
        :return: grad_x
        """
        # True during training, False during testing
        does_require_grad = x_est.requires_grad

        with torch.enable_grad():
            # Necessary for using autograd
            x_est.requires_grad_(True)
            # Assuming uniform white noise, in the denoising case matrix A is the identity
            error = torch.sum((y - x_est)**2)
            # We retain the graph during training only
            grad_x = torch.autograd.grad(error, inputs=x_est, retain_graph=does_require_grad,
                                         create_graph=does_require_grad)[0]
        # Set requires_grad back to it's original state
        x_est.requires_grad_(does_require_grad)

        return grad_x

    # At every iteration of the IRIM we use an Invertible Unet for processing. Note, that the use of ModuleList
    # is necessary for Pytorch to properly register all modules.
    step_models = torch.nn.ModuleList([InvertibleUnet(n_channels=n_channels,n_hidden=n_hidden,dilations=dilations,
                                       conv_nd=conv_nd, n_householder=n_householder) for i in range(n_steps)])

    # Build IRIM
    model = IRIM(step_models,grad_fun,im_channels)
    # Wrap the model to be trained with invert to learn
    model = MemoryFreeInvertibleModule(model)
    model.to(device)

    return model


class DataTransform:
    """
    Data Transformer for running U-Net models on a test dataset.
    """

    def __init__(self, resolution, which_challenge, mask_func=None):
        """
        Args:
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        
        if which_challenge not in ('singlecoil', 'multicoil'):
            #raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
            raise ValueError('Challenge should either be singlecoil or multicoil')
            print(which_challenge)
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.Array): k-space measurements
            target (numpy.Array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (pathlib.Path): Path to the input file
            slice (int): Serial number of the slice
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Normalized zero-filled input image
                mean (float): Mean of the zero-filled image
                std (float): Standard deviation of the zero-filled image
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = transforms.to_tensor(kspace)
        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, _ = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image to given resolution if larger
        smallest_width = min(args.resolution, image.shape[-2])
        smallest_height = min(args.resolution, image.shape[-3])
        image = transforms.complex_center_crop(image, (smallest_height, smallest_width))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image)
        image = image.clamp(-6, 6)
        return image, mean, std, fname, slice


def create_data_loaders(args):
    mask_func = None
    if args.mask_kspace:
        mask_func = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    data = SliceData(
        root=args.data_path / f'{args.challenge}_{args.data_split}',
        transform=DataTransform(args.resolution, args.challenge, mask_func),
        sample_rate=1.,
        challenge=args.challenge
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    if args.irim_train:
        model = make_IRIM()
    else:
        model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    print(model.load_state_dict(checkpoint['model']))
    return model
from tqdm import tqdm

def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames, slices) in tqdm(data_loader):
            if args.irim_train:
                input = input.unsqueeze(1).to(args.device)
                input_init = torch.cat((input,torch.zeros(input.size(0),n_channels[0]-im_channels,*[im_size]*conv_nd, device=args.device)),1)
                input_init.requires_grad_(False)
                recons = model.forward(input_init, input)[:,:im_channels].squeeze(1).to('cpu')
            else:
                input = input.unsqueeze(1).to(args.device)
                recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--mask-kspace', action='store_true',
                        help='Whether to apply a mask (set to True for val data and False '
                             'for test data')
    parser.add_argument('--data-split', choices=['val', 'test'], required=True,
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--irim-train', action='store_true',
                        help='If set, train with irim')
    return parser


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    
    args = create_arg_parser().parse_args(sys.argv[1:])
    
    # ---- Parameters ---
    # Working with images, for time series or volumes set to 1 or 3, respectively
    conv_nd = 2

    im_channels = 1
    # Number of total samples
    n_samples = args.batch_size
    im_size = 320
    #learning_rate = 1e-3
    device = args.device
    
    # Number of Householder projections for constructing 1x1 convolutions
    n_householder = 3
    # Number of channels for each layer of the Invertible Unet
    n_channels = [64] * 12
    # Number of hidden channel in the residual functions of the Invertible Unet
    n_hidden = [64, 64, 128, 128, 256, 1024, 1024, 256, 128, 128, 64, 64]
    # Downsampling factors
    dilations = [1, 1, 2, 2, 4, 8, 8, 4, 2, 2, 1, 1]
    # Number of IRIM steps]
    n_steps = 8
    
    
    main(args)
