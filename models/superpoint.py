# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

import numpy as np
import cv2
import matplotlib.cm as cm
from kornia.augmentation import RandomAffine, RandomPerspective
from kornia.geometry.transform import warp_perspective

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2*radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
            scores[:, None], width, 1, radius, divisor_override=1)
    ar = torch.arange(-radius, radius+1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(
            scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
            scores[:, None], kernel_x.transpose(2, 3), padding=radius)
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    Note(DD): Modified to perform inference time "homographic adaptation" to
    improve keypoint localization when combined with subpixel refinement.

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'refinement_radius': 0, # Subpixel refinement, 2 works well here.
        'adapt_num': 0, # Setting this to 1 or greater will enable HA.
        'adapt_seed': 0,
        'adapt_viz': False,
        'adapt_parallel': False,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        if self.config['adapt_num'] > 0:
            print('Running with \"adapt_num\" = {}'.format(self.config['adapt_num']))
            print('Running with \"adapt_seed\" = {}'.format(self.config['adapt_seed']))
            # Seed logic for warps. Different images should have different
            # warps yet be repeatable using adapt_seed.
            self.image_count = 0
            torch.random.manual_seed(self.config['adapt_seed'])
            self.initial_seed = torch.randint(high=999999, size=(1,))
            #torch.random.seed() # Restore generator.
            #torch.cuda.seed()
            # Random homography generator (composition of two transforms).
            self.aug1 = RandomPerspective(p=1.0,
                                          distortion_scale=0.5,
                                          return_transform=True)
            self.aug2 = RandomAffine(degrees=360,
                                     translate=(0.2, 0.2),
                                     scale=(0.5, 3.0),
                                     shear=(-10, 10),
                                     return_transform=True)
            if self.config['adapt_viz']:
                print('Homographic Adaptation viz mode ON')
                self.win = 'Homographic Adapt Viz'
                cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.win, (640, 480))

        print('Loaded SuperPoint model')


    def run_encoder(self, image):
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, 1, h*8, w*8)

        # Compute the dense descriptors
        x = x[0,...].unsqueeze(0) # Keep first descriptor in batch if multiple,
                                  # NOTE(dd): could be problematic if you are
                                  # running on batches of different images.
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        return scores, descriptors, h, w

    def viz_scores(self, scores, title):
        hm = scores[0,0,:,:].data.cpu().numpy()
        min_heat = 0.001
        hm[hm<min_heat] = min_heat
        hm = -np.log(hm)
        hm = (hm - hm.min()) / (hm.max() - hm.min())
        hm = (cm.jet(hm)*255.).astype('uint8')[:,:,:3]
        hm2 = hm.copy()
        ds = hm.shape[-2] / 640.
        cv2.putText(hm2, title, (5, int(ds*30)),
                    cv2.FONT_HERSHEY_SIMPLEX, ds*1.0, (0,0,0), 2, lineType=16)
        cv2.putText(hm2, title, (5, int(ds*30)),
                    cv2.FONT_HERSHEY_SIMPLEX, ds*1.0, (255,255,255), 1, lineType=16)
        return hm2

    def viz_image(self, image, title):
        im = (image[0,0,:,:].data.cpu().numpy()*255.).astype('uint8')
        im = np.dstack([im]*3)
        ds = im.shape[-2] / 640.
        cv2.putText(im, title, (5,int(ds*50)), cv2.FONT_HERSHEY_SIMPLEX, ds*1.0,
                    (0,0,0), 2, lineType=16)
        cv2.putText(im, title, (5,int(ds*50)), cv2.FONT_HERSHEY_SIMPLEX, ds*1.0,
                    (255,255,255), 1, lineType=16)
        return im

    def homographic_adapt(self, image):
        # Set a repeatable random seed that changes with a new image.
        torch.manual_seed(self.initial_seed + self.image_count)
        self.image_count += 1

        # Number of forward passes is 'adapt_num' + 1, since first is identity.
        adapt_num_plus_one = self.config['adapt_num'] + 1

        if self.config['adapt_parallel']:
            all_images = image.repeat(adapt_num_plus_one, 1, 1, 1)
            masks = torch.ones_like(all_images)
            all_images2, H1 = self.aug1(all_images)
            all_images2, H2 = self.aug2(all_images2)
            Hinv = torch.inverse(H2 @ H1)
            # First warp is always identity.
            all_images2[0,...] = image.clone()
            Hinv[0,...] = torch.eye(3)
            all_scores, descriptors, h, w = self.run_encoder(all_images2)
            masks_inv = warp_perspective(masks, Hinv, masks.shape[-2:])
            scores_inv = warp_perspective(all_scores, Hinv, all_scores.shape[-2:])
            scores_sum = scores_inv.sum(dim=0, keepdim=True)
            scores_count = masks_inv.sum(dim=0, keepdim=True)
            scores_mean = scores_sum / (scores_count + 1e-6)
        else:
            # Keep a running sum and count to compute mean later.
            scores_sum = torch.zeros_like(image)
            scores_count = torch.zeros_like(image)
            debug_images = []
            all_images = []
            all_scores = []
            for i in range(adapt_num_plus_one):
                mask = torch.ones_like(image)
                image2 = image.clone()
                if i == 0: # First warp is always identity.
                    scores, descriptors, h, w = self.run_encoder(image2)
                    Hinv = torch.eye(3)
                    mask_inv = mask
                    scores_inv = scores
                else:
                    image2, H1 = self.aug1(image2)
                    image2, H2 = self.aug2(image2)
                    Hinv = torch.inverse(H2 @ H1)
                    scores, _, _, _ = self.run_encoder(image2)
                    mask_inv = warp_perspective(mask, Hinv, mask.shape[-2:])
                    scores_inv = warp_perspective(scores, Hinv, scores.shape[-2:])
                scores_sum += scores_inv
                scores_count += mask_inv
                all_images.append(image2)
                all_scores.append(scores)
            scores_mean = scores_sum / (scores_count + 1e-6)
            all_images = torch.cat(all_images, dim=0)
            all_scores = torch.cat(all_scores, dim=0)

        # Restore the generator.
        if not image.is_cuda:
            torch.random.seed() # Crashes if call in GPU mode.
        torch.cuda.seed()

        if self.config['adapt_viz']:
            out = []
            viz_hm = self.viz_scores(scores_mean, 'Final Mean Scores')
            viz_im = self.viz_image(image, 'Original Image')
            out.append(np.hstack((viz_im, viz_hm)))
            for i, (sc, im) in enumerate(zip(all_scores, all_images)):
                viz_hm = self.viz_scores(sc.unsqueeze(0), 'Scores Round {}'.format(i))
                viz_im = self.viz_image(im.unsqueeze(0), 'Image Round {}'.format(i))
                out.append(np.hstack((viz_im, viz_hm)))
            debug_image = np.stack(out)
            debug_image = debug_image.reshape(-1, debug_image.shape[2], 3)
            debug_image_path = 'debug_ha_superpoint.png'
            print('Writing adapt viz image to \"{}\"'.format(debug_image_path))
            cv2.imwrite(debug_image_path, debug_image)
            cv2.imshow(self.win, debug_image)
            cv2.waitKey(1)

        return scores_mean, descriptors, h, w


    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """

        if self.config['adapt_num'] > 0:
            scores, descriptors, h, w = self.homographic_adapt(data['image'])
        else:
            scores, descriptors, h, w = self.run_encoder(data['image'])
        scores = scores.squeeze(1)

        full_scores = scores
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        if self.config['refinement_radius'] > 0:
            keypoints = soft_argmax_refinement(
                keypoints, full_scores, self.config['refinement_radius'])

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
