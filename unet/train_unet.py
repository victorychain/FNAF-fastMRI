"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common.args import Args
from common.subsample import create_mask_for_mask_type
from data import transforms
from data.mri_data import SliceData
from unet_model import UnetModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        target = transforms.to_tensor(target)
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        fn_image = 0
        if args.fn_train:
            #fn_image_kspace = kspace.clone()
            fn_image = transforms.ifft2(kspace).clone()
            # Crop input image to given resolution if larger
            smallest_width = min(min(args.resolution, fn_image.shape[-2]), target.shape[-1])
            smallest_height = min(min(args.resolution, fn_image.shape[-3]), target.shape[-2])
            crop_size = (smallest_height, smallest_width)
            fn_image = transforms.complex_center_crop(fn_image, crop_size)

            # Absolute value
            fn_image = transforms.complex_abs(fn_image)
            # Apply Root-Sum-of-Squares if multicoil data
            if self.which_challenge == 'multicoil':
                fn_image = transforms.root_sum_of_squares(fn_image)



        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image to given resolution if larger
        smallest_width = min(min(args.resolution, image.shape[-2]), target.shape[-1])
        smallest_height = min(min(args.resolution, image.shape[-3]), target.shape[-2])
        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)
        target = transforms.center_crop(target, crop_size)

        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        item = image, target, mean, std, attrs['norm'].astype(np.float32)
        if args.fn_train:
            item = image, target, mean, std, attrs['norm'].astype(np.float32), fn_image
        return item



def create_datasets(args):
    train_mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    dev_mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=args.data_path / f'{args.challenge}_train',
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
        bbox_root=args.bbox_root,
    )
    dev_data = SliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[123]]+[dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(gt - pred) ** 2 / torch.norm(gt) ** 2

def batch_nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.stack([nmse(gt[i], pred[i]) for i in range(len(gt))])

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    if args.fn_train:
        #loss_f = torch.nn.MSELoss(reduction='none')
        loss_f = torch.nn.L1Loss(reduction='none')
        mask_f = create_mask_for_mask_type('random',[0.08, 0.04],  [4, 8])
        
        n_pixel_range = (10, 100)
        n_pixel_range = (args.min_n_pixel, args.max_n_pixel)

    for iter, data in enumerate(data_loader):
        if args.bbox_root:
            (input, target, mean, std, norm), seg = data
        else:
            if not args.fn_train:
                input, target, mean, std, norm = data
            else:
                input, target, mean, std, norm, fn_image = data
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)


        output = model(input).squeeze(1)

        loss = F.l1_loss(output, target)

        if args.bbox_root:
            writer.add_scalar('L1_Loss', loss.item(), global_step + iter)

            
            bbox_loss = []
            for j in range(11):
                seg_mask = seg[:, :, :, j]
                if seg_mask.sum() > 0:
                    seg_mask = seg_mask.to(args.device)
                    bbox_output = output * seg_mask
                    bbox_target = target * seg_mask
                    bbox_loss.append(nmse(bbox_target, bbox_output))

            if len(bbox_loss)>0:
                bbox_loss = 2 * torch.stack(bbox_loss).mean()
                #print(loss.item(), bbox_loss.item())
                writer.add_scalar('BBOX_Loss', bbox_loss.item(), global_step + iter)
                loss += bbox_loss



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #information_loss_list, xs_list, ys_list = [], [], []
        
        if args.fn_train:
            #fn_attack_train(model, target, optimizer)
            #run_finite_diff(model, target, iterations=2, train=True, optimizer=optimizer)
            get_attack_loss_new(model, fn_image,
                           loss_f=loss_f, 
                                xs=np.random.randint(low=from_boarder, high=320-from_boarder, size=(fn_image.size(0),)), 
                               ys=np.random.randint(low=from_boarder, high=320-from_boarder, size=(fn_image.size(0),)), 
                                shape=(320, 320), n_pixel_range=n_pixel_range, train=True, optimizer=optimizer)

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()

    loss_f = batch_nmse

    n_pixel_range = (10, 11)

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            if not args.fn_train:
                input, target, mean, std, norm = data
            else:
                input, target, mean, std, norm, fn_image = data

            if not args.fn_train:
                input = input.unsqueeze(1).to(args.device)
                target = target.to(args.device)

                output = model(input).squeeze(1)

                mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
                std = std.unsqueeze(1).unsqueeze(2).to(args.device)
                target = target * std + mean
                output = output * std + mean

                norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
                loss = F.mse_loss(output / norm, target / norm, size_average=False)
                losses.append(loss.item())
            else:

                fnaf_loss_list = []
                for _ in range(11):
                    fnaf_loss = get_attack_loss_new(model, fn_image,
                                   loss_f=loss_f, 
                                        xs=np.random.randint(low=from_boarder, high=320-from_boarder, size=(fn_image.size(0),)), 
                                       ys=np.random.randint(low=from_boarder, high=320-from_boarder, size=(fn_image.size(0),)), 
                                        shape=(320, 320), n_pixel_range=n_pixel_range, train=False, optimizer=None)

                    fnaf_loss_list.append(fnaf_loss.cpu().numpy())

                fnaf_loss = np.max(fnaf_loss_list, axis=0)
                print(fnaf_loss)
                losses += list(fnaf_loss)
        if not args.fn_train:
            writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
        else:
            return losses
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            if not args.fn_train:
                input, target, mean, std, norm = data
            else:
                input, target, mean, std, norm, fn_image = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)


            output = model(input)
            #save_image(input, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    #args = checkpoint['args']

    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    print(model.load_state_dict(checkpoint['model']))

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    if not args.new_optimizer:
        optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def k_space_to_image_with_mask(kspace, mask_func=None, seed=None):
    #use_seed = False
    #seed = None if not use_seed else tuple(map(ord, fname))
    #seed = 42
    #print(fname)
    #kspace = transforms.to_tensor(kspace)
    if mask_func:
        masked_kspace, mask = transforms.apply_mask(kspace, mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
    else:
        image = transforms.ifft2(kspace)
    image = transforms.complex_abs(image)
    image = transforms.center_crop(image, (320, 320))
    # Normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    return image

def to_k_space(image):
    #image = image.numpy()
    image = np.complex64(image)
    image = transforms.to_tensor(image)
    return transforms.fft2(image)
    
def fn_attack_train(model, target, optimizer):

    loss_f = torch.nn.MSELoss()

    perturb_noise = perturb_noise_init(n_pixel_range=(10, 11))
            
    # perturb the target to get the perturbed image
    #perturb_noise = np.expand_dims(perturb_noise, axis=0)
    #perturb_noise = np.stack((perturb_noise,)*target.shape(0), 0)
    perturb_noise = [perturb_noise_init(n_pixel_range=(10, 11)) for _ in range(target.size(0))]
    perturb_noise = np.stack(perturb_noise, 0) 

    new_target = target.detach().cpu().numpy() + perturb_noise
    target_k_space = to_k_space(new_target)
    input = k_space_to_image_with_mask(target_k_space, mask_func=mask_f).unsqueeze(1).to(args.device)

    # apply the perturbed image to the model to get the loss

    output = model(input).squeeze(1)

    perturb_noise_tensor = transforms.to_tensor(perturb_noise).to(args.device)

    mask = adjusted_mask((perturb_noise > 0).astype(np.double), d=0.995)
    #mask = torch.tensor(perturb_noise_tensor, requires_grad=False)
    #mask[mask > 0] = 1
    mask = transforms.to_tensor(mask).to(args.device, dtype=torch.double)

    #loss = loss_f(output*mask, perturb_noise_tensor)*100
    loss = loss_f(output*mask, transforms.to_tensor(new_target).to(args.device, dtype=torch.double)*mask)*100

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
from scipy.ndimage import distance_transform_edt as distance
def dis(mask):
    bg = 1-mask
    bg_edt = distance(bg)
    bg_edt2 = (np.max(bg_edt)-bg_edt) * bg
    return bg_edt2 /np.max(bg_edt2)

def adjusted_mask(mask, d=0.999):
    #return ((dis(mask)>d).astype(np.double) + mask).astype(np.double)
    return (LCC(dis(mask)>d) + mask).astype(np.double)

from scipy import ndimage
def LCC(binary_img):
    og_img = binary_img.copy()
    binary_img = binary_img > 0
    label_im, nb_labels = ndimage.label(binary_img)
    sizes = ndimage.sum(binary_img, label_im, range(nb_labels + 1))
    max_size = np.max(sizes)
    mask_size = sizes < max_size
    remove_pixel = mask_size[label_im]
    og_img[remove_pixel] = 0
    return og_img


def fn_attack_batch(model, images, n_repeat=10, prior_fn=None, eps=0.3, alpha=2/255, iters=1):
    #images = images.to(device)
    #labels = labels.to(device)
    #loss_f = torch.nn.L1Loss(reduction='mean')
    loss_f = torch.nn.MSELoss(reduction='none')

    #mask_f = create_mask_for_mask_type('random',[0.08, 0.04],  [4, 8])

    ori_image = images.data
    #print(ori_target.shape)

    loss_list = []
    information_loss_list, xs_list, ys_list = [], [], []
    
    n_pixel_range = (100, 50000)

    for _ in range(n_repeat):
        
            loss = get_attack_loss_new(model, ori_target, information_loss_list, xs_list, ys_list,
                           loss_f=loss_f, 
                                xs=np.random.randint(low=100, high=320-100, size=(ori_image.size(0),)), 
                               ys=np.random.randint(low=100, high=320-100, size=(ori_image.size(0),)), 
                                shape=(320, 320), n_pixel_range=n_pixel_range)
            loss_list.append(loss)

    loss_list = np.stack(loss_list, axis=0)
    final_loss = np.max(loss_list, axis=0)
    #print(final_loss)
    return final_loss

def fn_evaluate(args, model, data_loader, epoch, writer, n_repeat=10):
    model.eval()
    losses = np.array([])
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            #if iter == 0:
            #    continue
            input, target, mean, std, norm, fn_image = data
            input = input.unsqueeze(1).to(args.device)
            target = target.to(args.device)

            loss = fn_attack_batch(model, fn_image, n_repeat=n_repeat)
            losses = np.concatenate([losses, loss])


            #output = model(input).squeeze(1)

            #mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            #std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            #target = target * std + mean
            #output = output * std + mean

            #norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            #loss = F.mse_loss(output / norm, target / norm, size_average=False)
            #losses.append(loss.item())
            #return
        #writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    
    fn_rate = np.count_nonzero((losses) >= 0.0010) / len(losses)
    writer.add_scalar('FN_Rate', fn_rate, epoch)
    writer.add_scalar('FN_Loss', np.mean(losses), epoch)
    print('fn_rate:', fn_rate)
    print('fn_mean_loss:', np.mean(losses))
    return losses, fn_rate

UPDATE_PARAMS = ['x', 'y']
LRS = {'x': 1e7, 'y': 1e7}
def set_param(param, val, params):
    params[param] = val



def approx_partial(model, ori_target, param, current_val, params, loss_list, information_loss_list, xs_list, ys_list, train=False, optimizer=None):
    """Compute the approximate partial derivative using the finite-difference method.
    :param param:
    :param current_val:
    :param params:
    :return:
    """
    #step_size = STEP_SIZES[param]
    step_size = 10
    losses = []

    for sign in [-1, 1]:
        set_param(param, current_val + sign * step_size / 2, params)
        loss = get_attack_loss(model, ori_target, information_loss_list, xs_list, ys_list,
                               loss_f=torch.nn.MSELoss(reduction='none'), 
                                xs=params['x'], ys=params['y'], 
                                shape=(320, 320), n_pixel_range=(10, 11), train=train, optimizer=optimizer)
        # image = RENDERER.render()
        # with torch.no_grad():
        #     out = MODEL(image)

        # loss = CRITERION(out, LABELS).item()
        losses.append(loss)

    grad = (losses[1] - losses[0]) / step_size
    loss_list += losses
    return grad

def run_finite_diff(model, ori_target, iterations=2, train=False, optimizer=None):

    params = {'x': np.random.randint(low=100, high=320-100, size=(ori_target.size(0),)), 
              'y': np.random.randint(low=100, high=320-100, size=(ori_target.size(0),))}
    loss_list = []
    
    information_loss_list, xs_list, ys_list = [], [], []
    
    
    
    loss_f=torch.nn.MSELoss(reduction='none')

    loss = get_attack_loss(model, ori_target, information_loss_list, xs_list, ys_list,
                           loss_f=loss_f, 
                                xs=params['x'], ys=params['y'], 
                                shape=(320, 320), n_pixel_range=(10, 11), train=train, optimizer=optimizer)
    loss_list.append(loss)

    for current_iter in range(iterations):
        grads = {
            param: approx_partial(model, ori_target, param, params[param], params, 
                                  loss_list, information_loss_list, xs_list, ys_list, train=train, optimizer=optimizer)
            for param in UPDATE_PARAMS
        }
        
        #print(grads)

        for param in grads:

            val = params[param]
            val += LRS[param] * grads[param]

            params[param] = val

        loss = get_attack_loss(model, ori_target, information_loss_list, xs_list, ys_list,
                               loss_f=loss_f, 
                                xs=params['x'], ys=params['y'], 
                                shape=(320, 320), n_pixel_range=(10, 11), train=train, optimizer=optimizer)
        loss_list.append(loss)
        
        
    loss_list = np.stack(loss_list, axis=0)
    
    index = np.argmax(loss_list, axis=0)
    #print(index)
    information_loss_list, xs_list, ys_list = np.stack(information_loss_list), np.stack(xs_list), np.stack(ys_list)
    information_loss = information_loss_list[index, np.arange(len(index))]
    xs = xs_list[index, np.arange(len(index))]
    ys = ys_list[index, np.arange(len(index))]
    
    
    return np.max(loss_list, axis=0), information_loss, xs, ys

def fd_evaluate(args, model, data_loader, writer, epoch, iterations=2):
    model.eval()
    losses = np.array([])
    information_losses, xses, yses = np.array([]), np.array([]), np.array([])
    
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            #if iter == 0:
            #    continue
            input, target, mean, std, norm, fn_image = data
            input = input.unsqueeze(1).to(args.device)
            target = target.to(args.device)

            #loss = fn_attack_batch(model, input, target, n_repeat=n_repeat)
            loss, information_loss, xs, ys = run_finite_diff(model, target, iterations=iterations)
            losses = np.concatenate([losses, loss])
            information_losses, xses, yses = np.concatenate([information_losses, information_loss]), np.concatenate([xses, xs]), np.concatenate([yses, ys])
            print(loss, information_loss, xs, ys)


            #output = model(input).squeeze(1)

            #mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            #std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            #target = target * std + mean
            #output = output * std + mean

            #norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            #loss = F.mse_loss(output / norm, target / norm, size_average=False)
            #losses.append(loss.item())
            #return
        #writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    
    fn_rate= np.count_nonzero((losses) >= 0.0010) / len(losses)
    info_rate= np.count_nonzero((information_losses) >= 0.00010) / len(information_losses)
    print('fn_rate:', fn_rate)
    print('fn_mean_loss:', np.mean(losses))
    print('info_rate:', info_rate)
    print('info_mean_loss:', np.mean(information_losses))
    writer.add_scalar('FN_Rate', fn_rate, epoch)
    writer.add_scalar('FN_Loss', np.mean(losses), epoch)
    writer.add_scalar('Info_Rate', info_rate, epoch)
    return losses, information_losses, xses, yses

def get_attack_loss_new(model, ori_target, loss_f=torch.nn.MSELoss(reduction='none'), 
    xs=np.random.randint(low=100, high=320-100, size=(16,)), 
    ys=np.random.randint(low=100, high=320-100, size=(16,)), 
    shape=(320, 320), n_pixel_range=(10, 11), train=False, optimizer=None):
    
    input_o = ori_target.unsqueeze(1).to(args.device)
    input_o = input_o.clone()
    
    #input_o = transforms.complex_abs(ori_input.clone())
    #input_o, mean, std = transforms.normalize_instance(ori_target.unsqueeze(1).clone())
    #input_o = torch.clamp(input_o, -6, 6)

    #perturb_noise = perturb_noise_init(x=x, y=y, shape=shape, n_pixel_range=n_pixel_range)
    p_max = input_o.max().cpu()
    #p_min = (p_max - input.min()) / 2
    #p_min = (p_max - input_o.min())
    p_min = input_o.min().cpu()
    perturb_noise = [perturb_noise_init(x=x, y=y, shape=shape, n_pixel_range=n_pixel_range, pixel_value_range=(p_min, p_max)) for x, y in zip(xs, ys)]
    perturb_noise = np.stack(perturb_noise)
            
    # perturb the target to get the perturbed image
    #perturb_noise = np.expand_dims(perturb_noise, axis=0)
    #perturb_noise = np.stack((perturb_noise,)*ori_target.shape(0), -1)

    seed = np.random.randint(999999999)
    
    
    perturb_noise = transforms.to_tensor(perturb_noise).unsqueeze(1).to(args.device)
    
    if not args.fnaf_eval_control:
        input_o += perturb_noise
    target = input_o.clone()
    
    #print(input_o.shape)
    input_o = np.complex64(input_o.cpu().numpy())
    input_o = transforms.to_tensor(input_o)
    input_o = transforms.fft2(input_o)
    input_o, mask = transforms.apply_mask(input_o, mask_f, seed)
    input_o = transforms.ifft2(input_o)
    
    image = transforms.complex_abs(input_o).to(args.device)
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    
    target = transforms.normalize(target, mean, std, eps=1e-11)
    target = target.clamp(-6, 6)

    #information_loss = loss_f(og_image.squeeze(1), image.squeeze(1)).mean(-1).mean(-1).cpu().numpy()
    #information_loss = np.array([0]*len(xs))

    # apply the perturbed image to the model to get the loss
    if train:
        output = model(image).squeeze(1)
    else:
        with torch.no_grad():
            output = model(image).squeeze(1)
            
    #perturb_noise_tensor = transforms.to_tensor(perturb_noise).to(args.device, dtype=torch.double)
    perturb_noise = perturb_noise.squeeze(1)
    perturb_noise_tensor = perturb_noise
    
    perturb_noise = perturb_noise.cpu().numpy()
        
    mask = adjusted_mask((perturb_noise > 0).astype(np.double))
    #mask = (perturb_noise > 0).astype(np.double)
    

        
    target = target.squeeze(1)
    mask_0 = transforms.to_tensor(mask).to(args.device)

    loss = loss_f((output*mask_0), (target*mask_0))

    if train:
        b_loss = loss.sum() / mask_0.sum() * 1 + loss_f(output, target).mean()
        b_loss.backward()
        optimizer.step()
        loss = loss.detach()

        loss = loss.mean(-1).mean(-1).cpu().numpy()
    #loss = loss.mean(-1).mean(-1).numpy()

    # information_loss_list.append(information_loss)
    # xs_list.append(xs)
    # ys_list.append(ys)
    
    
    return loss

from_boarder = 50
def perturb_noise_init(x=np.random.randint(low=100, high=320-100), y=np.random.randint(low=100, high=320-100), 
    shape=(320, 320), n_pixel_range=(10, 11), pixel_value_range=(3, 6)):
    
    if shape[0] > 320:

        x = int(np.clip(x, 100+24, 320-(100+24)))
        y = int(np.clip(y, 100+24, 320-(100+24)))
    else:
        x = int(np.clip(x, from_boarder, 320-(from_boarder)))
        y = int(np.clip(y, from_boarder, 320-(from_boarder)))

    image = np.zeros(shape, dtype=np.float32)

    for _ in range(np.random.randint(low=n_pixel_range[0], high=n_pixel_range[1])):

        image[x, y] = np.random.uniform(low=pixel_value_range[0], high=pixel_value_range[1], size=(1))

        if np.random.choice([-1, 1]) > 0:
            #x = np.clip(x + np.random.choice([-1, 1]), -shape[0], shape[0]-1)
            if shape[0] > 320:
                x = np.clip(x + np.random.choice([-1, 1]), 100+24, 320-(100+24))
            else:
                x = np.clip(x + np.random.choice([-1, 1]), from_boarder, 320-(from_boarder))
        else:
            #y = np.clip(y + np.random.choice([-1, 1]), -shape[1], shape[1]-1)
            if shape[0] > 320:
                y = np.clip(y + np.random.choice([-1, 1]), 100+24, 320-(100+24))
            else:
                y = np.clip(y + np.random.choice([-1, 1]), from_boarder, 320-(from_boarder))

#     for i in range(np.random.randint(low=n_pixel_range[0], high=n_pixel_range[1])):

#         image[x, y] = list0[i]

#         if list1[i] > 0:
#             x = np.clip(x + list2[i], -shape[0], shape[0]-1)
#         else:
#             y = np.clip(y + list3[i], -shape[1], shape[1]-1)

    return image



def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        if args.new_optimizer:
            optimizer = build_optim(args, model.parameters())
        #args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']+10
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        if args.best_model:
            
            _, model, optimizer = load_model(args.best_model)
            #print(model)
        else:

            model = build_model(args)

        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    #logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    #if not args.new_optimizer:
    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    #else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    if args.fnaf_eval:
        fnaf_losses = evaluate(args, 0, model, dev_loader, writer)
        if args.fnaf_eval_control:
            control = '_control'
        else:
            control = ''
        np.save(args.fnaf_eval/"fnaf_losses{}.npy".format(control), fnaf_losses)
    else:
        for epoch in range(start_epoch, args.num_epochs):
            #scheduler.step(epoch)
            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
            
            dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
            visualize(args, epoch, model, display_loader, writer)
            #if not args.new_optimizer:
            #    scheduler.step(epoch)
            #else:
            scheduler.step(dev_loss)
            
            #if args.fn_train:
                #fn_evaluate(args, model, dev_loader, epoch, writer, n_repeat=11)
                #fd_evaluate(args, model, dev_loader, writer, epoch, iterations=2)

            is_new_best = dev_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss, dev_loss)
            save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--fn-train', action='store_true',
                        help='If set, train with fn')

  
    parser.add_argument('--best-model', type=str,
                        help='Path to an existing best model checkpoint.')
    
    parser.add_argument('--gpu', type=str,
                        help='Which gpu to use')
    
    parser.add_argument('--max_n_pixel', type=float, default=100.,
                        help='n_pixel max')
    
    parser.add_argument('--min_n_pixel', type=float, default=10.,
                        help='n_pixel min')
    
    parser.add_argument('--new_optimizer', type=int, default=1,
                        help='new_optimizer')

    parser.add_argument('--bbox_root', type=str, default=None)

    parser.add_argument('--fnaf_eval', type=pathlib.Path, default=None,
                        help='Path where fnaf eval results should be saved')

    parser.add_argument('--fnaf_eval_control', action='store_true')

    
    return parser


if __name__ == '__main__':

    
    args = create_arg_parser().parse_args()
                        
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu 
                        
    mask_f = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
                        
                      


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
