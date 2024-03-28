import time
import os
# from colored

class Config():
    nowtime = time.strftime("%Y%m%d-%H%M")

    model_name = ['UNet', 'UNetLK', 'UXNet', 'unetr', 'swin_unetr'][4]
    input_mode = ['Crop', 'Cube'][0]
    # dataset parameters
    batch_size       = 1
    crop_size = (1024, 32, 64)                                  ###
    target_shape = (1024, 32, 256)
    num_samples_per_cube = 3

    num_workers = 4

    # model parameters
    in_chans             = 1
    num_classes          = 1    

    # training setting
    init_lr       = 5e-3
    end_lr        = 5e-4
    epochs_bg     = 106                                             ###
    epochs_end    = 1000
    model_weights = None
    if epochs_bg > 0: 
        nowtime = ['20240314-1753', "20240311-1134", '20240321-1033'
            ][2]
        model_weights = f'model_weights-{epochs_bg:04d}'
    path_tb     = f"./data/log/{nowtime}/tensorboard"
    path_weight = f"./data/log/{nowtime}/weights"
    path_demo = f"./data/log/{nowtime}/demo"
    

    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    # model_weights = [None, '20221222-1345'][0]
    # if model_weights is not None:
    #     model_weights = f"./data/log/{nowtime}/weights"

    print(f"\033[32m{model_name} {'====='*4} Building training process from {nowtime}\033[0m")


    

