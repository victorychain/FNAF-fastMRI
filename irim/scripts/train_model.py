"""
This code is based on the training code found at
https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
"""

import sys, os
import gc
import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch

import torchvision
from torch.utils.tensorboard import SummaryWriter

from irim.rim import RIM, ConvRNN
from irim import IRIM, InvertibleUnet, ResidualBlockPixelshuffle

from fastMRI.common import evaluate as numpy_eval
from fastMRI.common.args import Args
from fastMRI.data import transforms

from training_utils.models import IRIMfastMRI, RIMfastMRI
from training_utils.data_loaders import create_training_loaders
from training_utils.helpers import mse_gradient, estimate_to_image, image_loss, real_to_complex, complex_to_real

from training_utils.mri_data import SliceData
from training_utils.data_transformers import TrainingTransform, TestingTransform
from torch.utils.data import DataLoader
from fastMRI.common.subsample import create_mask_for_mask_type




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
from scipy.ndimage import distance_transform_edt as distance
def dis(mask):
    bg = 1-mask
    bg_edt = distance(bg)
    bg_edt2 = (np.max(bg_edt)-bg_edt) * bg
    return bg_edt2 /np.max(bg_edt2)
def adjusted_mask(mask, d=0.999):
    return (LCC(dis(mask)>d) + mask).astype(np.double)
    #return ((dis(mask)>d).astype(np.double)).astype(np.double)

def perturb_noise_init(x=np.random.randint(low=100, high=320-100), y=np.random.randint(low=100, high=320-100), 
    shape=(320, 320), n_pixel_range=(10, 11), pixel_value_range=(3, 6)):
    
    if shape[0] > 320:

        x = int(np.clip(x, 100+24, 320-(100+24)))
        y = int(np.clip(y, 100+24, 320-(100+24)))
    else:
        x = int(np.clip(x, 100, 320-(100)))
        y = int(np.clip(y, 100, 320-(100)))

    image = np.zeros(shape, dtype=np.float32)

    for _ in range(np.random.randint(low=n_pixel_range[0], high=n_pixel_range[1])):

        image[x, y] = np.random.uniform(low=pixel_value_range[0], high=pixel_value_range[1], size=(1))

        if np.random.choice([-1, 1]) > 0:
            #x = np.clip(x + np.random.choice([-1, 1]), -shape[0], shape[0]-1)
            if shape[0] > 320:
                x = np.clip(x + np.random.choice([-1, 1]), 100+24, 320-(100+24))
            else:
                x = np.clip(x + np.random.choice([-1, 1]), 100, 320-(100))
        else:
            #y = np.clip(y + np.random.choice([-1, 1]), -shape[1], shape[1]-1)
            if shape[0] > 320:
                y = np.clip(y + np.random.choice([-1, 1]), 100+24, 320-(100+24))
            else:
                y = np.clip(y + np.random.choice([-1, 1]), 100, 320-(100))

    return image

def get_attack_loss(args, model, ori_target, fnaf_mask, loss_f=torch.nn.MSELoss(reduction='none'), 
    xs=np.random.randint(low=100, high=320-100), 
    ys=np.random.randint(low=100, high=320-100), 
    shape=(320, 320), n_pixel_range=(10, 11), vis=False):
    
    ori_target, metadata, target_norm, ori_input = ori_target

    input_o = transforms.complex_abs(ori_input.clone())


    
    p_max = input_o.max()
    #p_min = (p_max - input.min()) / 2
    #p_min = (p_max - input_o.min())
    p_min = (input_o.min())
    perturb_noise = [perturb_noise_init(x=x, y=y, shape=shape, n_pixel_range=n_pixel_range, pixel_value_range=(p_min, p_max)) for x, y in zip(xs, ys)]
    perturb_noise = np.stack(perturb_noise)
            
    # perturb the target to get the perturbed image
    #perturb_noise = np.expand_dims(perturb_noise, axis=0)
    #perturb_noise = np.stack((perturb_noise,)*ori_target.shape(0), -1)

    seed = np.random.randint(9999999)
    
    
    

    # normalizer = target_norm
    # for i in range(len(ori_target.size()) - 1):
    #     normalizer = normalizer.unsqueeze(-1)
            
    #target = ori_target / normalizer
    #print('target: ', target.max(), target.min())
    
    #perturb_noise = torch.stack([transforms.to_tensor(perturb_noise).unsqueeze(1)]*2, -1)
    perturb_noise = transforms.to_tensor(perturb_noise).unsqueeze(1)
    
    if not args.fnaf_eval_control:
        input_o += perturb_noise
    target = input_o.clone()
    
    #print(input_o.shape)
    input_o = np.complex64(input_o.numpy())
    input_o = transforms.to_tensor(input_o)
    input_o = transforms.fft2(input_o)
    input_o, mask = transforms.apply_mask(input_o, fnaf_mask, seed=seed)
    input_o = transforms.ifft2(input_o)

    # apply the perturbed image to the model to get the loss
    #print(input_o.shape)
    output = model.forward(y=input_o, mask=mask, metadata=metadata)
    #output = torch.zeros((8, 1, 368, 368, 2)).to(args.device)
    output = estimate_to_image(output, args.resolution)
    #output = output.reshape(-1, 1, output.size(-2), output.size(-1)).squeeze(1)
    
#             output /= normalizer.cuda()
    #output, _, _ = transforms.normalize_instance(output, eps=1e-11)
    
    #output = transforms.normalize(output, mean, std, eps=1e-11)
    #output = output.clamp(-6, 6)
            
    
    #perturb_noise_tensor = transforms.to_tensor(perturb_noise).to(args.device, dtype=torch.double)
    perturb_noise = torch.stack([perturb_noise]*2, -1)
    perturb_noise = estimate_to_image(perturb_noise, args.resolution).numpy()

    mask = adjusted_mask((perturb_noise != 0))
    #mask = (perturb_noise > 0).astype(np.double)
  

    mask = transforms.to_tensor(mask).to(args.device)
    
    #loss = loss_f((output.cpu()*mask_0), (transforms.to_tensor(target)*mask_0))
    target = torch.stack([target]*2, -1)
    target = estimate_to_image(target, args.resolution).to(args.device)
#     target /= normalizer
    #target, _, _ = transforms.normalize_instance(target, eps=1e-11)
#     target = target.clamp(-6, 6)
    #target = transforms.normalize(output, mean, std, eps=1e-11)
    
    #loss = loss_f(target*mask, output*mask).sum() / torch.sum(mask)
    loss = loss_f(target*mask, output*mask)
    
    #loss = loss.mean(-1).mean(-1).cpu().numpy()
    #loss = loss.mean(-1).mean(-1).numpy()
    
    if vis and loss.max() >= 0.001:
        print('vis!')
        print(output.min(), output.max())
        print(target.min(), target.max())
        
    
    return loss

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(gt - pred) ** 2 / torch.norm(gt) ** 2

def batch_nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.stack([nmse(gt[i], pred[i]) for i in range(len(gt))])

def train_epoch(args, epoch, model, train_loader, optimizer, writer, fnaf_iterloader=None, fnaf_loader=None, fnaf_mask=None):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(train_loader)

    #loss_f = torch.nn.MSELoss(reduction='none')
    loss_f = nmse

    memory_allocated = []
    if args.fnaf_train:
        loader = zip(train_loader, fnaf_loader)
    else:
        loader = train_loader
    for i, data in enumerate(loader):
        if args.fnaf_train:
            data, batch = data
        if args.bbox_root:
            (y, mask, target, metadata, target_norm, target_max), seg = data
        else:
            y, mask, target, metadata, target_norm, target_max = data
        y = y.to(args.device)
        mask = mask.to(args.device)
        target = target.to(args.device)
        target_norm = target_norm.to(args.device)
        target_max = target_max.to(args.device)

        optimizer.zero_grad()
        model.zero_grad()
        estimate = model.forward(y=y, mask=mask, metadata=metadata)
        if isinstance(estimate, list):
            loss = [image_loss(e, target, args, target_norm, target_max) for e in estimate]
            loss = sum(loss) / len(loss)
        else:
            loss = image_loss(estimate, target, args, target_norm, target_max)

        if args.bbox_root:
            writer.add_scalar('SSIM_Loss', loss.item(), global_step + i)

            
            bbox_loss = []
            for j in range(11):
                seg_mask = seg[:, :, :, j]
                if seg_mask.sum() > 0:
                    seg_mask = seg_mask.to(args.device)
                    bbox_output = estimate_to_image(estimate, args.resolution) * seg_mask
                    bbox_target = target * seg_mask
                    bbox_loss.append(nmse(bbox_target, bbox_output))

            if bbox_loss:
                bbox_loss = 10 * torch.stack(bbox_loss).mean()
                #print(loss.item(), bbox_loss.item())
                writer.add_scalar('BBOX_Loss', bbox_loss.item(), global_step + i)
                loss += bbox_loss



        loss.backward()
        optimizer.step()



        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if i > 0 else loss.item()
        writer.add_scalar('Loss', loss.item(), global_step + i)


        if args.fnaf_train:
            optimizer.zero_grad()
            model.zero_grad()
            y, mask, target, metadata, target_norm, target_max = batch
            labels = target, metadata, target_norm, y
            # fnaf_loss = get_attack_loss(model, labels,
            #                loss_f=loss_f, 
            #                     xs=np.random.randint(low=100, high=320-100, size=(ori_image.size(0),)), 
            #                    ys=np.random.randint(low=100, high=320-100, size=(ori_image.size(0),)), 
            #                     shape=(320, 320), n_pixel_range=(10, 11), vis=vis)
            vis = False
            #n_pixel_range = (10, 11)
            #n_pixel_range = (10, 101)
            #n_pixel_range = (10, 1001)
            n_pixel_range = (10, 5001)
            fnaf_loss = get_attack_loss(args, model, labels,
                           loss_f=loss_f, 
                                xs=np.random.randint(low=100+24, high=368-(100+24), size=(y.size(0),)), 
                               ys=np.random.randint(low=100+24, high=368-(100+24), size=(y.size(0),)), 
                                shape=(368, 368), n_pixel_range=n_pixel_range, fnaf_mask=fnaf_mask, vis=vis)
            
            #fnaf_loss = 10000000 * fnaf_loss
            fnaf_loss = 10 * fnaf_loss
            #print(loss, fnaf_loss)
            writer.add_scalar('FNAF_Loss', fnaf_loss.item(), global_step + i)

            fnaf_loss.backward()
            optimizer.step()

        if args.device == 'cuda':
            memory_allocated.append(torch.cuda.max_memory_allocated() * 1e-6)
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.empty_cache()
        gc.collect()

        if i % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{i:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.detach().item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s '
                f'Memory allocated (MB) = {np.min(memory_allocated):.2f}'
            )
            memory_allocated = []
        start_iter = time.perf_counter()
    optimizer.zero_grad()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, fnaf_mask=None):
    model.eval()
    mse_losses = []
    psnr_losses = []
    nmse_losses = []
    ssim_losses = []
    memory_allocated = []

    fnaf_losses = []

    start = time.perf_counter()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if not args.fnaf_eval:
                y, mask, target, metadata = data[:4]
                y = y.to(args.device)
                mask = mask.to(args.device)
                target = target.to(args.device)
                if args.n_slices > 1:
                    output = model.forward(y=y, mask=mask, metadata=metadata)
                    output = estimate_to_image(output, args.resolution)
                    output_np = output.to('cpu').transpose(0, -4).squeeze(-4)
                    del output
                else:
                    y = y.transpose(0, -4).squeeze(-4)
                    mask = mask.squeeze(-4).repeat(y.size(0), 1, 1, 1, 1)
                    metadata = metadata.repeat(y.size(0), 1)
                    output_np = []
                    for k, l in zip(range(0, y.size(0), args.batch_size),
                                    range(args.batch_size, y.size(0) + args.batch_size, args.batch_size)):
                        output = model.forward(y=y[k:l], mask=mask[k:l], metadata=metadata[k:l])
                        output = estimate_to_image(output, args.resolution)
                        output_np.append(output.to('cpu'))
                    output_np = torch.cat(output_np, 0)

                output_np = output_np.reshape(-1, output_np.size(-2), output_np.size(-1))
                target = target.reshape_as(output_np)

                output_np = output_np.to('cpu').numpy()
                target_np = target.to('cpu').numpy()
                mse_losses.append(numpy_eval.mse(target_np, output_np))
                psnr_losses.append(numpy_eval.psnr(target_np, output_np))
                nmse_losses.append(numpy_eval.nmse(target_np, output_np))
                ssim_losses.append(numpy_eval.ssim(target_np, output_np))

            if args.fnaf_eval:
                loss_f = batch_nmse

                y, mask, target, metadata, target_norm, target_max = data
                labels = target, metadata, target_norm, y
                # fnaf_loss = get_attack_loss(model, labels,
                #                loss_f=loss_f, 
                #                     xs=np.random.randint(low=100, high=320-100, size=(ori_image.size(0),)), 
                #                    ys=np.random.randint(low=100, high=320-100, size=(ori_image.size(0),)), 
                #                     shape=(320, 320), n_pixel_range=(10, 11), vis=vis)
                vis = False
                n_pixel_range = (10, 11)
                #n_pixel_range = (10, 101)
                #n_pixel_range = (10, 1001)
                #n_pixel_range = (10, 5001)

                fnaf_loss_list = []

                for _ in range(11):
                    fnaf_loss = get_attack_loss(args, model, labels,
                                   loss_f=loss_f, 
                                        xs=np.random.randint(low=100+24, high=368-(100+24), size=(y.size(0),)), 
                                       ys=np.random.randint(low=100+24, high=368-(100+24), size=(y.size(0),)), 
                                        shape=(368, 368), n_pixel_range=n_pixel_range, fnaf_mask=fnaf_mask, vis=vis)

                    fnaf_loss_list.append(fnaf_loss.cpu().numpy())

                fnaf_loss = np.max(fnaf_loss_list, axis=0)
                #print(fnaf_loss)
                fnaf_losses += list(fnaf_loss)

        if not args.fnaf_eval:
            writer.add_scalar('Val_MSE', np.mean(mse_losses), epoch)
            writer.add_scalar('Val_PSNR', np.mean(psnr_losses), epoch)
            writer.add_scalar('Val_NMSE', np.mean(nmse_losses), epoch)
            writer.add_scalar('Val_SSIM', np.mean(ssim_losses), epoch)
            writer.add_scalar('Val_memory', np.max(memory_allocated), epoch)
    if args.fnaf_eval:
        out = fnaf_losses
    else:
        out = np.mean(nmse_losses), np.mean(psnr_losses), np.mean(mse_losses), np.mean(ssim_losses), \
           time.perf_counter() - start, np.max(memory_allocated)

    return out


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    output_images = []
    target_images = []
    corrupted_images = []
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            y, mask, target, metadata = data[:4]
            y = y.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)

            n_slices = target.size(-3)
            # corrupted = transforms.root_sum_of_squares(estimate_to_image(
            #     y[..., n_slices // 2, :, :, :], target.size()[-2:]), 1)
            if args.n_slices == 1:
                mask = mask.squeeze(-4)
                y = y[..., n_slices // 2, :, :, :]

            estimate = model.forward(y=y, mask=mask, metadata=metadata)
            estimate.detach_()

            target = target[..., n_slices // 2, :, :]
            #target_norm = target.norm(dim=(-2, -1), keepdim=True)
            # corrupted_images.append(corrupted / target_norm)
            # target_images.append(target / target_norm)
            #corrupted_images.append(corrupted)
            target_images.append(target)

            if args.n_slices > 1:
                # output_images.append(
                #     estimate_to_image(estimate[..., n_slices // 2, :, :, :],
                #                       target.size()[-2:]).clone().detach() / target_norm)
                output_images.append(
                    estimate_to_image(estimate[..., n_slices // 2, :, :, :],
                                      target.size()[-2:]).clone().detach())
            else:
                #output_images.append(estimate_to_image(estimate, target.size()[-2:]).clone().detach() / target_norm)
                output_images.append(estimate_to_image(estimate, target.size()[-2:]).clone().detach())

    output = torch.cat(output_images, 0)[:16].unsqueeze(1)
    target = torch.cat(target_images, 0)[:16].unsqueeze(1)
    #corrupted = torch.cat(corrupted_images, 0)[:16].unsqueeze(1)

    #print(corrupted.shape, target.shape)
    save_image(target, 'Target')
    #save_image(corrupted, 'Corrupted')
    save_image(output, 'Reconstruction')
    save_image(target - output, 'Error')
    #save_image(corrupted - output, 'Corrupted_Reconstruction_Difference')
    #save_image(corrupted - target, 'Corrupted_Target_Difference')


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
    im_channels = 2 if args.challenge == 'singlecoil' else 30
    if args.use_rim:
        conv_nd = 3 if args.n_slices > 1 else 2
        rnn = ConvRNN(2 * im_channels, conv_dim=conv_nd)
        model = RIM(rnn, grad_fun=mse_gradient)
        model = RIMfastMRI(model, n_steps=args.n_steps)
    else:
        im_channels = im_channels * args.multiplicity
        channels = args.n_hidden if isinstance(args.n_hidden, list) else [args.n_hidden] * args.depth
        assert channels[0] >= 2 * im_channels + 8
        dilations = args.dilations
        n_hidden = args.n_network_hidden
        conv_nd = 3 if args.n_slices > 1 else 2
        if conv_nd == 3:
            # Make sure to not downsample in the slice direction
            dilations = [[1, d, d] for d in dilations]
        if args.shared_weights:
            cell_ = InvertibleUnet(n_channels=channels, n_hidden=n_hidden, dilations=dilations, conv_nd=conv_nd)
            cell = torch.nn.ModuleList([cell_] * args.n_steps)
        else:
            cell = torch.nn.ModuleList([InvertibleUnet(n_channels=channels, n_hidden=n_hidden,
                                                       dilations=dilations, conv_nd=conv_nd)
                                        for i in range(args.n_steps)])
        if args.parametric_output:
            output_function = ResidualBlockPixelshuffle(channels[0], 2, channels[0], conv_nd=conv_nd, use_glu=False)
        else:
            output_function = lambda x: complex_to_real(real_to_complex(x)[:,:im_channels // (2 * args.multiplicity)])

        model = IRIM(cell, grad_fun=mse_gradient, n_channels=im_channels)
        model = IRIMfastMRI(model, output_function, channels[0], multiplicity=args.multiplicity)

    return model.to(args.device)


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    print(model.load_state_dict(checkpoint['model']))

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    if args.optimizer.upper() == 'RMSPROP':
        optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'ADAM':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    if args.optimizer.upper() == 'SGD':
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    checkpoint_pretrained = os.path.join(args.exp_dir, 'pretrained.pt')
    if args.checkpoint is None:
        checkpoint_path = os.path.join(args.exp_dir, 'model.pt')
    else:
        checkpoint_path = args.checkpoint

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint, model, optimizer = load_model(checkpoint_path)
        #args = checkpoint['args']
        #best_dev_loss = checkpoint['best_dev_loss']
        print('Best dev loss', checkpoint['best_dev_loss'])
        best_dev_loss = 1e9
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        if os.path.exists(checkpoint_pretrained):
            _, model, optimizer = load_model(checkpoint_pretrained)
            optimizer.lr = args.lr
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    #logging.info(model)

    train_loader, val_loader, display_loader = create_training_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)



    

    if args.fnaf_train:

        fnaf_data = SliceData(
            root=args.data_path / f'{args.challenge}_train',
            transform=TrainingTransform(None, args.resolution, args.challenge, use_seed=True,
                                    train_resolution=args.train_resolution),
            sample_rate=args.sample_rate,
            challenge=args.challenge,
            n_slices=args.n_slices,
            use_rss=args.use_rss
        )

        fnaf_loader = DataLoader(
            dataset=fnaf_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        fnaf_iterloader = iter(fnaf_loader)

        #fnaf_mask = MaskFunc(args.center_fractions, args.accelerations)
        fnaf_mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    else:
        fnaf_loader = None
        fnaf_iterloader = None
        fnaf_mask = None



    if args.fnaf_eval:

        fnaf_val_data = SliceData(
            root=args.data_path / f'{args.challenge}_val',
            transform=TrainingTransform(None, args.resolution, args.challenge, use_seed=True,
                                        train_resolution=args.train_resolution),
            sample_rate=args.sample_rate,
            challenge=args.challenge,
            n_slices=args.n_slices,
            use_rss=args.use_rss
        )

        #fnaf_val_data = [fnaf_val_data[i] for i in range(24*2)]

        fnaf_val_loader = DataLoader(
            dataset=fnaf_val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        #fnaf_mask = MaskFunc(args.center_fractions, args.accelerations)
        fnaf_mask = create_mask_for_mask_type(args.mask_type, args.val_center_fractions, args.val_accelerations)

    else:
        fnaf_mask = None


    if args.fnaf_eval:
        fnaf_losses = evaluate(args, 0, model, fnaf_val_loader, writer, fnaf_mask=fnaf_mask)
        if args.fnaf_eval_control:
            control = '_control'
        else:
            control = ''
        np.save(args.fnaf_eval/"fnaf_losses{}.npy".format(control), fnaf_losses)
    else:
        for epoch in range(start_epoch, args.num_epochs):
            visualize(args, epoch, model, display_loader, writer)
            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer, fnaf_iterloader=fnaf_iterloader, fnaf_loader=fnaf_loader, fnaf_mask=fnaf_mask)
            nmse_loss, psnr_loss, mse_loss, ssim_loss, dev_time, dev_mem = evaluate(args, epoch, model, val_loader, writer)
            scheduler.step(epoch)

            ssim_loss = -ssim_loss
            #ssim_loss = nmse_loss
            is_new_best = ssim_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss, ssim_loss)
            save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'VAL_NMSE = {nmse_loss:.4g} VAL_MSE = {mse_loss:.4g} VAL_PSNR = {psnr_loss:.4g} '
                f'VAL_SSIM = {ssim_loss:.4g} \n'
                f'TrainTime = {train_time:.4f}s ValTime = {dev_time:.4f}s ValMemory = {dev_mem:.2f}',
            )
            if args.exit_after_checkpoint:
                writer.close()
                sys.exit(0)
    writer.close()


def create_arg_parser():
    parser = Args()
    # Mask parameters
    parser.add_argument('--val_accelerations', nargs='+', default=[8], type=int)
    parser.add_argument('--val_center_fractions', nargs='+', default=[0.04], type=float)
    parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--use_rim', action='store_true',
                        help='If set, RIM with fixed parameters')
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer to use choose between"
                                                                      "['Adam', 'SGD', 'RMSProp']")
    parser.add_argument('--loss', choices=['l1', 'mse', 'nmse', 'ssim'], default='ssim', help='Training loss')
    parser.add_argument('--loss_subsample', type=float, default=1., help='Sampling rate for loss mask')
    parser.add_argument('--use_rss', action='store_true',
                        help='If set, will train singlecoil model with RSS targets')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_slices', type=int, default=1, help='Number of slices in an observation. Default=1, if'
                                                                'n_slices > 1, we will use 3d convolutions')
    parser.add_argument('--lr_step_size', type=int, default=40, help='Period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--n_steps', type=int, default=8, help='Number of RIM steps')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report_interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data_parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--multiplicity', type=int, default=1,
                        help='Number of eta estimates at every time step. The higher multiplicity, the lower the '
                             'number of necessary time steps would be expected.')
    parser.add_argument('--shared_weights', action='store_true',
                        help='If set, weights will be shared over time steps. (only relevant for IRIM)')
    parser.add_argument('--n_hidden', type=int, nargs='+', help='Number of hidden features in each layer. Can'
                                                                'be either Int or List of Ints')
    parser.add_argument('--n_network_hidden', type=int, nargs='+', help='Number of hidden features in each layer. Can'
                                                                        'be either Int or List of Ints')
    parser.add_argument('--dilations', type=int, nargs='+', help='Kernel dilations in each in each layer. Can'
                                                                 'be either Int or List of Ints')
    parser.add_argument('--depth', type=int, help='Number of RNN layers.')
    parser.add_argument('--train_resolution', type=int, nargs=2, default=None, help='Image resolution during training')
    parser.add_argument('--parametric_output', action='store_true', help='Use a parametric function for map the'
                                                                         'last layer of the iRIM to an image estimate')
    parser.add_argument('--exit_after_checkpoint', action='store_true')

    parser.add_argument('--fnaf_train', action='store_true')
    parser.add_argument('--fnaf_eval', type=pathlib.Path, default=None,
                        help='Path where fnaf eval results should be saved')
    parser.add_argument('--fnaf_eval_control', action='store_true')
    parser.add_argument('--bbox_root', type=str, default=None)
    parser.add_argument('--gpu', type=str, help='GPU')

    return parser


if __name__ == '__main__':

    args = create_arg_parser().parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu 

    torch.cuda.current_device()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
