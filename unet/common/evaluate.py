"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tqdm import tqdm
from skimage.transform import resize
from sewar.full_ref import vifp, ssim, msssim
#from vif import vifvec

from PIL import Image
# import warnings
# warnings.filterwarnings("error")
import matplotlib.pyplot as plt

from copy import deepcopy
import os


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )

def bbox_ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    # size = 100
    # gt = resize(gt, (size, size))
    # pred = resize(pred, (size, size))
    return structural_similarity(
        gt, pred, multichannel=False, data_range=gt.max()
    )

def bbox_vifp(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    size = 100
    gt = resize(gt, (size, size))
    pred = resize(pred, gt.shape)
    return vifp(
        gt, pred
    )

def bbox_mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.sum((gt - pred) ** 2) / np.count_nonzero(gt)


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)




class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }
        self.metrics_values = {
            metric: [[] for _ in range(11)] for metric in metric_funcs
        }
        self.metric_funcs = metric_funcs

    def push(self, target, recons, index=None):
        for metric, func in self.metric_funcs.items():
            v = func(target, recons)
            self.metrics[metric].push(v)
            if index is not None:
                self.metrics_values[metric][index].append(v)
            else:
                self.metrics_values[metric].append(v)

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def plot(self):
        
        for metric, stat in self.metrics_values.items():
            plt.figure()
            print(metric, len(stat))         
            plt.hist(stat, bins=len(stat))
            plt.savefig('{}_{}_hist.png'.format("1", metric))
        

    def __repr__(self):
        #self.plot()
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )

def get_segment_crop(img, mask):
    #return img[np.ix_(mask.any(1), mask.any(0))]
    return img[mask > 0]

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

def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    if args.bbox_root:
        if args.by_slice:
            BBOX_METRIC_FUNCS = dict(
                BBOX_MSE=bbox_mse,
                BBOX_NMSE=nmse,
                #BBOX_SSIM=bbox_ssim,
                #BBOX_VIPF=bbox_vifp,
            )
        else:
            BBOX_METRIC_FUNCS = dict(
                BBOX_MSE=bbox_mse,
                BBOX_NMSE=nmse,
            )
        bbox_metrics = Metrics(BBOX_METRIC_FUNCS)
        #bbox_metrics = [Metrics(BBOX_METRIC_FUNCS) for _ in range(11)]

        bbox_size_list = [[] for _ in range(11)]

    n = -1
    for tgt_file in tqdm(args.target_path.iterdir()):
        n += 1
        # if n in [67, 119, 143, 157, 161] :
        #     continue
        # if n != 161:
        #     continue
        # if n > 10:
        #     break
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            if args.acquisition and args.acquisition != target.attrs['acquisition']:
                continue


            if args.bbox_root:
                seg_fname = args.bbox_root+tgt_file.name[:-2]+'cii'
                #seg_fname = '/data/knee_mri8/Francesco/fastMRI/fastMRI_bb/singlecoil_val/bb/check_maskBB/'+tgt_file.name[:-2]+'cii'
                try:
                    seg = h5py.File(seg_fname, 'r')['mask'][()]
                except:
                    continue

            target = target[recons_key][()]
            recons = recons['reconstruction'][()]
            if not args.bbox_root:       
                metrics.push(target, recons)

            if args.bbox_root:

                seg = np.flip(seg, 0)
                #seg = np.moveaxis(seg, 2, 0)
                seg = seg.transpose(2, 0, 1, 3)

                if args.by_slice:
                    for i in range(seg.shape[0]):
                        seg_slice = seg[i]
                        target_slice = target[i]
                        recon_slice = recons[i]
                        for j in range(11):
                            seg_mask = seg_slice[:, :, j]
                            if np.sum(seg_mask) > 0:

                                # if np.sum(seg_mask) != np.sum(LCC(seg_mask)):
                                #     # print(seg_fname, i, j)
                                #     # im = Image.fromarray((seg_mask * 255).astype(np.uint8))
                                #     # im.save("two_mask.png")
                                #     print(np.sum(seg_mask), np.sum(LCC(seg_mask)))
                                try:
                                    seg_mask = seg_mask.astype(np.float)
                                    bbox_metrics.push(get_segment_crop(target_slice, mask=seg_mask), 
                                                get_segment_crop(recon_slice, mask=seg_mask), index=j)
                                    bbox_size_list[j].append(np.sum(seg_mask))
                                except Exception as e:
                                    print(n, e)
                                    print(get_segment_crop(target_slice, mask=seg_mask).shape)

                                    continue

                else:
                    for j in range(11):
                        seg_mask = seg[:, :, :, j]
                        if np.sum(seg_mask) > 0:
                            #bbox_metrics.push(target*seg_mask, recons*seg_mask)
                            #bbox_size_list.append(np.sum(seg_mask))
                            bbox_metrics[j].push(target*seg_mask, recons*seg_mask, index=j)


    if args.bbox_root:
        metrics = bbox_metrics

        #np.save("bbox_size_list.npy", np.array(bbox_size_list))
        np.save("bbox_size_by_slice_list.npy", bbox_size_list)

    if args.bbox_root:
        if args.by_slice:
            np.save(args.predictions_path / "bbox_stats_by_slice.npy", metrics.metrics_values)
        else:
            np.save(args.predictions_path / "bbox_stats.npy", metrics.metrics_values)
    else:
        np.save(args.predictions_path / "stats.npy", metrics.metrics_values)


    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                        help='Which challenge')
    parser.add_argument('--acquisition', choices=['CORPD_FBK', 'CORPDFS_FBK', 'AXT1', 'AXT1PRE',
                        'AXT1POST', 'AXT2', 'AXFLAIR'], default=None,
                        help='If set, only volumes of the specified acquisition type are used '
                             'for evaluation. By default, all volumes are included.')

    parser.add_argument('--bbox-root', type=str, default=None,
                        help='Root directory for the bbox masks')
    parser.add_argument('--by-slice', action='store_true',
                        help='use slice as a unit instead of volume for bbox masks')

    args = parser.parse_args()

    recons_key = 'reconstruction_rss' if args.challenge == 'multicoil' else 'reconstruction_esc'

    og_args = deepcopy(args)
    for directory in os.listdir(args.predictions_path):
        if os.path.isdir(args.predictions_path / directory):
            print(directory)
            args.predictions_path = og_args.predictions_path / directory
            metrics = evaluate(args, recons_key)
            print(metrics)



