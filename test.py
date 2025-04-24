import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import esrt

# Testing settings

parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_hr_folder", type=str, default='Test_Datasets/Set5/',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='Test_Datasets/Set5_LR/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/Set5/x2')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_x2.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)


# def forward_chop(model, x, shave=10, min_size=60000):
#     scale = 4 #self.scale[self.idx_scale]
#     n_GPUs = 1 #min(self.n_GPUs, 4)
#     b, c, h, w = x.size()
#     h_half, w_half = h // 2, w // 2
#     h_size, w_size = h_half + shave, w_half + shave
#     lr_list = [
#         x[:, :, 0:h_size, 0:w_size],
#         x[:, :, 0:h_size, (w - w_size):w],
#         x[:, :, (h - h_size):h, 0:w_size],
#         x[:, :, (h - h_size):h, (w - w_size):w]
#     ]

#     if w_size * h_size < min_size:
#         sr_list = []
#         for i in range(0, 4, n_GPUs):
#             lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
#             sr_batch = model(lr_batch)
#             sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
#     else:
#         sr_list = [
#             forward_chop(model, patch, shave=shave, min_size=min_size) \
#             for patch in lr_list
#         ]

#     h, w = scale * h, scale * w
#     h_half, w_half = scale * h_half, scale * w_half
#     h_size, w_size = scale * h_size, scale * w_size
#     shave *= scale

#     output = x.new(b, c, h, w)
#     output[:, :, 0:h_half, 0:w_half] \
#         = sr_list[0][:, :, 0:h_half, 0:w_half]
#     output[:, :, 0:h_half, w_half:w] \
#         = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
#     output[:, :, h_half:h, 0:w_half] \
#         = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
#     output[:, :, h_half:h, w_half:w] \
#         = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

#     return output

def forward_chop(model, x, scale, shave=10, min_size=60000):
    """Modified forward_chop that handles different scales better"""
    try:
        n_GPUs = 1
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        # Make sure h_size and w_size don't exceed the image dimensions
        h_size = min(h, h_size)
        w_size = min(w, w_size)

        # Create the four patches
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                forward_chop(model, patch, scale=scale, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        # Scale up dimensions
        h_out, w_out = scale * h, scale * w
        h_half_out, w_half_out = scale * h_half, scale * w_half
        h_size_out, w_size_out = scale * h_size, scale * w_size

        # Create a new tensor directly instead of using x.new()
        output = torch.zeros(b, c, h_out, w_out, device=x.device)

        # Ensure all dimensions are valid
        h_half_out = min(h_half_out, sr_list[0].size(2))
        w_half_out = min(w_half_out, sr_list[0].size(3))

        # For top-left patch (0)
        output[:, :, 0:h_half_out, 0:w_half_out] = sr_list[0][:, :, 0:h_half_out, 0:w_half_out]

        # For top-right patch (1)
        w_size_right = min(w_size_out, sr_list[1].size(3))
        right_offset = max(0, w_size_out - (w_out - w_half_out))
        right_slice_width = min(w_out - w_half_out, sr_list[1].size(3) - right_offset)

        output[:, :, 0:h_half_out, w_half_out:w_half_out + right_slice_width] = \
            sr_list[1][:, :, 0:h_half_out, right_offset:right_offset + right_slice_width]

        # For bottom-left patch (2)
        h_size_bottom = min(h_size_out, sr_list[2].size(2))
        bottom_offset = max(0, h_size_out - (h_out - h_half_out))
        bottom_slice_height = min(h_out - h_half_out, sr_list[2].size(2) - bottom_offset)

        output[:, :, h_half_out:h_half_out + bottom_slice_height, 0:w_half_out] = \
            sr_list[2][:, :, bottom_offset:bottom_offset + bottom_slice_height, 0:w_half_out]

        # For bottom-right patch (3)
        # Ensure we don't go out of bounds with carefully calculated slice dimensions
        bottom_right_h = min(h_out - h_half_out, sr_list[3].size(2) - bottom_offset)
        bottom_right_w = min(w_out - w_half_out, sr_list[3].size(3) - right_offset)

        output[:, :, h_half_out:h_half_out + bottom_right_h, w_half_out:w_half_out + bottom_right_w] = \
            sr_list[3][:, :, bottom_offset:bottom_offset + bottom_right_h, right_offset:right_offset + bottom_right_w]

        return output
    except Exception as e:
        print(f"Error in forward_chop: {str(e)}")
        print(f"Input tensor size: {x.size()}, scale: {scale}")
        # Fallback to direct forward pass if chopping fails
        return model(x)


cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

scale = opt.upscale_factor
filepath = opt.test_hr_folder
lr_filepath = opt.test_lr_folder
# if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
#     ext = '.bmp'
# else:
ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = esrt.ESRT(upscale=opt.upscale_factor)  #
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=False)  # True)

print("Loaded model successfully!")

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


def get_lr_image_path(hr_image):
    lr_image_path = lr_filepath + "/" + hr_image.split('/')[-1].split('.')[0].replace("HR", "LR") + ext
    return lr_image_path


for index, imname in enumerate(filelist):
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    print(imname)
    im_l = cv2.imread(get_lr_image_path(imname), cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = forward_chop(model, im_input, scale=scale)  # model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)

    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1

print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
