from monai.networks.nets.swin_unetr import SwinUNETR

if __name__ == "__main__":
    import os, torch
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    from torchkeras import summary
    net = SwinUNETR(in_channels=1, out_channels=3, img_size=(512,32,64), norm_name='instance')
    summary(net, torch.randn((2,1,992,32,256)))
