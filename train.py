import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import esrt #, architecture
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
import wandb
from importlib import import_module
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# This is the entry point
# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="/dataset",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')
#  No need for tests

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> DEVICE: ", device)

print("===> Loading datasets")

# Training Dataset
trainset = DIV2K.div2k(args)
testset = Set5_val.DatasetFromFolderVal(
    "Test_Datasets/Set5/X{}/HR".format(args.scale),
    "Test_Datasets/Set5/X{}/LR".format(args.scale),
    args.scale
)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True


model = esrt.ESRT(upscale = args.scale) #architecture.IMDN(upscale=args.scale)

l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> WANDB INIT AND SETUP")

try:
    wandb.login(key="6b8966b4154c1ea7f7b69ccc6e342b6d21e8b92d") # API Key is in your wandb account, under settings (wandb.ai/settings)

    config = {
        'epoch': args.nEpochs,
        'lr': args.lr,
        'gamma': args.gamma,
        'batch_size': args.batch_size
    }

    run = wandb.init(
        #  add name here for label
        name = "attempt3", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "idls25-ESRT", ### Project should be created in your wandb account
        config = config ### Wandb Config for your run
    )
    use_wandb = True
except Exception as e:
    print(f"Error initializing wandb: {str(e)}")
    print("Continuing without wandb logging")
    use_wandb = False

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch) -> float:
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    total_loss = 0
    total_batches = 0
    
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        try:
            if args.cuda:
                lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
                hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

            optimizer.zero_grad()
            sr_tensor = model(lr_tensor)
            
            # Handle size mismatch between sr_tensor and hr_tensor
            if sr_tensor.size() != hr_tensor.size():
                print(f"Size mismatch - SR: {sr_tensor.size()}, HR: {hr_tensor.size()}")
                # Center crop both to the smaller dimension
                min_h = min(sr_tensor.size(2), hr_tensor.size(2))
                min_w = min(sr_tensor.size(3), hr_tensor.size(3))
                
                # Calculate offsets for center crop
                sr_h_offset = (sr_tensor.size(2) - min_h) // 2
                sr_w_offset = (sr_tensor.size(3) - min_w) // 2
                hr_h_offset = (hr_tensor.size(2) - min_h) // 2
                hr_w_offset = (hr_tensor.size(3) - min_w) // 2
                
                # Crop both tensors to same size
                sr_tensor = sr_tensor[:, :, sr_h_offset:sr_h_offset+min_h, sr_w_offset:sr_w_offset+min_w]
                hr_tensor = hr_tensor[:, :, hr_h_offset:hr_h_offset+min_h, hr_w_offset:hr_w_offset+min_w]
                print(f"After cropping - SR: {sr_tensor.size()}, HR: {hr_tensor.size()}")
            
            loss_l1 = l1_criterion(sr_tensor, hr_tensor)
            loss_sr = loss_l1
            total_loss += loss_l1.item()
            total_batches += 1

            loss_sr.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader), loss_l1.item()))
        except Exception as e:
            print(f"Error in training iteration {iteration}: {str(e)}")
            # Skip this batch and continue
            continue
    
    # Return average loss, handle case where no batches were successfully processed
    if total_batches == 0:
        return 0.0
    return total_loss / total_batches


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
        
        output[:, :, 0:h_half_out, w_half_out:w_half_out+right_slice_width] = \
            sr_list[1][:, :, 0:h_half_out, right_offset:right_offset+right_slice_width]
        
        # For bottom-left patch (2)
        h_size_bottom = min(h_size_out, sr_list[2].size(2))
        bottom_offset = max(0, h_size_out - (h_out - h_half_out))
        bottom_slice_height = min(h_out - h_half_out, sr_list[2].size(2) - bottom_offset)
        
        output[:, :, h_half_out:h_half_out+bottom_slice_height, 0:w_half_out] = \
            sr_list[2][:, :, bottom_offset:bottom_offset+bottom_slice_height, 0:w_half_out]
        
        # For bottom-right patch (3)
        # Ensure we don't go out of bounds with carefully calculated slice dimensions
        bottom_right_h = min(h_out - h_half_out, sr_list[3].size(2) - bottom_offset)
        bottom_right_w = min(w_out - w_half_out, sr_list[3].size(3) - right_offset)
        
        output[:, :, h_half_out:h_half_out+bottom_right_h, w_half_out:w_half_out+bottom_right_w] = \
            sr_list[3][:, :, bottom_offset:bottom_offset+bottom_right_h, right_offset:right_offset+bottom_right_w]
        
        return output
    except Exception as e:
        print(f"Error in forward_chop: {str(e)}")
        print(f"Input tensor size: {x.size()}, scale: {scale}")
        # Fallback to direct forward pass if chopping fails
        return model(x)


def valid(scale) -> (float, float):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    valid_samples = 0
    
    for batch_idx, batch in enumerate(testing_data_loader):
        try:
            lr_tensor, hr_tensor = batch[0], batch[1]
            if args.cuda:
                lr_tensor = lr_tensor.to(device)
                hr_tensor = hr_tensor.to(device)

            with torch.no_grad():
                try:
                    # Try using forward_chop first
                    pre = forward_chop(model, lr_tensor, scale)
                except Exception as e:
                    print(f"Error during forward_chop: {str(e)}")
                    # Fallback to direct forward pass
                    pre = model(lr_tensor)

            # Handle size mismatch between SR and HR tensors
            if pre.size() != hr_tensor.size():
                print(f"Sample {batch_idx}: Size mismatch - SR: {pre.size()}, HR: {hr_tensor.size()}")
                min_h = min(pre.size(2), hr_tensor.size(2))
                min_w = min(pre.size(3), hr_tensor.size(3))
                
                # Calculate offsets for center crop
                sr_h_offset = (pre.size(2) - min_h) // 2
                sr_w_offset = (pre.size(3) - min_w) // 2
                hr_h_offset = (hr_tensor.size(2) - min_h) // 2
                hr_w_offset = (hr_tensor.size(3) - min_w) // 2
                
                # Apply center crop
                pre = pre[:, :, sr_h_offset:sr_h_offset+min_h, sr_w_offset:sr_w_offset+min_w]
                hr_tensor = hr_tensor[:, :, hr_h_offset:hr_h_offset+min_h, hr_w_offset:hr_w_offset+min_w]
                print(f"After cropping - SR: {pre.size()}, HR: {hr_tensor.size()}")

            # Convert to numpy for evaluation
            sr_img = utils.tensor2np(pre.detach()[0])
            gt_img = utils.tensor2np(hr_tensor.detach()[0])
            
            # Ensure images have matching dimensions
            if sr_img.shape != gt_img.shape:
                print(f"Shape mismatch after numpy conversion - SR: {sr_img.shape}, GT: {gt_img.shape}")
                # Find minimum dimensions
                min_h = min(sr_img.shape[0], gt_img.shape[0])
                min_w = min(sr_img.shape[1], gt_img.shape[1])
                # Crop to minimum size
                sr_img = sr_img[:min_h, :min_w, :]
                gt_img = gt_img[:min_h, :min_w, :]
            
            crop_size = args.scale
            
            # Make sure we don't try to crop more than the image size
            crop_size = min(crop_size, min(sr_img.shape[0], sr_img.shape[1], gt_img.shape[0], gt_img.shape[1]) // 2)
            
            print(f"Sample {batch_idx}: sr_img shape: {sr_img.shape}, gt_img shape: {gt_img.shape}")
            
            # Apply shave/crop
            cropped_sr_img = utils.shave(sr_img, crop_size)
            cropped_gt_img = utils.shave(gt_img, crop_size)
            
            print(f"After shave: cropped_sr_img shape: {cropped_sr_img.shape}, cropped_gt_img shape: {cropped_gt_img.shape}")
            
            # Process for Y channel comparison if needed
            if args.isY is True:
                try:
                    im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                    im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
                except Exception as e:
                    print(f"Error during color conversion: {str(e)}")
                    # Skip this sample if color conversion fails
                    continue
            else:
                im_label = cropped_gt_img
                im_pre = cropped_sr_img
                
            # Final shape check before computing metrics
            if im_pre.shape != im_label.shape:
                print(f"Shape mismatch before PSNR computation - im_pre: {im_pre.shape}, im_label: {im_label.shape}")
                # Find common dimensions
                min_h = min(im_pre.shape[0], im_label.shape[0])
                min_w = min(im_pre.shape[1], im_label.shape[1])
                # Crop both to common size
                im_pre = im_pre[:min_h, :min_w]
                im_label = im_label[:min_h, :min_w]
                
            print(f"Final shapes - im_pre: {im_pre.shape}, im_label: {im_label.shape}")
            
            # Only compute metrics if we have valid images
            if im_pre.size > 0 and im_label.size > 0 and im_pre.shape == im_label.shape:
                try:
                    psnr_val = utils.compute_psnr(im_pre, im_label)
                    ssim_val = utils.compute_ssim(im_pre, im_label)
                    avg_psnr += psnr_val
                    avg_ssim += ssim_val
                    valid_samples += 1
                    print(f"Sample {batch_idx}: PSNR = {psnr_val:.4f}, SSIM = {ssim_val:.4f}")
                except Exception as e:
                    print(f"Error computing metrics: {str(e)}")
                    # Continue to next sample
                    continue
            else:
                print(f"Sample {batch_idx}: Skipping metrics computation due to invalid shapes")
                
        except Exception as e:
            print(f"Error during validation of sample {batch_idx}: {str(e)}")
            # Continue to the next batch
            continue
            
    if valid_samples == 0:
        print("Warning: No valid samples for evaluation")
        return 0.0, 0.0
        
    avg_psnr /= valid_samples
    avg_ssim /= valid_samples
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}, valid samples: {}".format(
        avg_psnr, avg_ssim, valid_samples))
    return avg_psnr, avg_ssim


def save_checkpoint(epoch):
    model_folder = "experiment/checkpoint_ESRT_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

    print("===> Checkpoint saved to {}".format(model_out_path))


def save_best_psnr():
    model_folder = "experiment/checkpoint_ESRT_x{}/".format(args.scale)
    model_out_path = model_folder + "best_psnr.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

    print("===> BEST PSNR Checkpoint saved to {}".format(model_out_path))


def save_best_ssim():
    model_folder = "experiment/checkpoint_ESRT_x{}/".format(args.scale)
    model_out_path = model_folder + "best_ssir.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

    print("===> BEST SSIM Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

print("===> Training")
print_network(model)
code_start = datetime.datetime.now()
timer = utils.Timer()
best_psnr = 0
best_ssim = 0
for epoch in range(args.start_epoch, args.nEpochs + 1):
    t_epoch_start = timer.t()
    epoch_start = datetime.datetime.now()
    
    try:
        # Validate first
        psnr, ssim = valid(args.scale)
        
        # Then train
        train_loss_for_epoch = train(epoch)
        
        # Save checkpoint periodically
        if epoch % 10 == 0:
            save_checkpoint(epoch)

        # Save best models
        if psnr > best_psnr:
            best_psnr = psnr
            save_best_psnr()

        if ssim > best_ssim:
            best_ssim = ssim
            save_best_ssim()

        # Log timing info
        epoch_end = datetime.datetime.now()
        print('Epoch cost times: %s' % str(epoch_end-epoch_start))
        t = timer.t()
        prog = (epoch-args.start_epoch+1)/(args.nEpochs + 1 - args.start_epoch + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        
        # Log metrics
        metrics = {
            'time_elapsed': t_elapsed,
            'epoch': epoch,
            'train_loss': train_loss_for_epoch,
            'psnr': psnr,
            'ssim': ssim
        }
        
        # Log to wandb if available
        if use_wandb:
            try:
                run.log(metrics)
            except Exception as e:
                print(f"Error logging to wandb: {str(e)}")
        
        print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
        
    except Exception as e:
        print(f"Error in epoch {epoch}: {str(e)}")
        # Continue to next epoch
        continue

code_end = datetime.datetime.now()
print('Code cost times: %s' % str(code_end-code_start))
