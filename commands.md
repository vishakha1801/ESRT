## Inference Command
``` 
python3 ESRT/test.py --upscale_factor 3 --checkpoint /ocean/projects/cis250019p/ndas/experiment/checkpoint_ESRT_x3/last_epoch.pth --output_folder . --test_lr_folder /ocean/projects/cis250019p/ndas/experiment/checkpoint_ESRT_x3/last_epoch.pth --test_hr_folder /ocean/projects/cis250019p/ndas/ESRT/Test_Datasets/Set5/X3/HR --test_lr_folder /ocean/projects/cis250019p/ndas/ESRT/Test_Datasets/Set5/X3/LR
```

## Train command
```
python3 ESRT/train.py --scale 3 --batch_size 20 --patch_size 92 --ext .png
```

## Start Training
``` 
python3 ESRT/train.py --scale 3 --batch_size 20 --patch_size 92 --ext .png --start-epoch 970 --pretrained /ocean/projects/cis250019p/ndas/experiment/checkpoint_ESRT_x3/last_epoch.pth
```
