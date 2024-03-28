from networks.unet_lk import UNetLK
from networks.unet import UNet
from networks.uxnet_3d import UXNET
import os
import glob
import torch 
import torch.nn as nn
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
from astropy.io import fits
import torchio as tio

class Inference:
    def __init__(
            self, 
            model_path: str,
            path_to_save,
            model_type
            ):
        self.init_model(model_path, model_type)
        self.path_to_save = path_to_save

    def init_model(self, model_path, model_type):
        if model_type == 'UNetLK':
            model = UNetLK(1,1,32)
        elif model_type == 'UXNet':
            model = UXNET(
                        in_chans=1,
                        out_chans=1,
                        depths=[2,2,2,2],
                        feat_size=[16,32,64,128],
                        drop_path_rate=0,
                        layer_scale_init_value=0.,
                        hidden_size=768,
                        spatial_dims=3
                        )
        model.eval()
        model_weights = torch.load(model_path)
        model.load_state_dict(model_weights)
        self.model = nn.DataParallel(model).cuda()

    def predict(self, cube_path_list):
        for cube_path in tqdm(cube_path_list):
            label = self.predict_single_cube(cube_path)



    def predict_single_cube(self, cube_path):
        img_cube, origin_shape = self.preprocessing(cube_path)
        with torch.no_grad():
            input_cube = img_cube.cuda().float()
            pred = self.model(input_cube).sigmoid().detach().cpu()[0,0]
        label = self.postprocessing(pred, origin_shape)
        
        label = tio.LabelMap(tensor=label.int())
        label_path = f"{self.path_to_save}/{cube_path.split('/')[-1]}"
        # label = tio.ScalarImage(tensor=torch.as_tensor(label[None]))
        label.save(label_path.replace('.fits', '-pred-label.nii.gz'))
        # image = tio.ScalarImage(tensor=torch.as_tensor(img_cube[0]))
        # image.save(label_path.replace('.fits', '-pred-image.nii.gz'))
        return label

    def preprocessing(self, cube_path):
        hdu = fits.open(cube_path)
        image = hdu[0].data.astype(np.float16)
        image = torch.as_tensor(image).float()
        origin_shape = image.shape
        image = torch.nn.functional.avg_pool3d(image[None, None], kernel_size=(6,1,1), stride=(4,1,1))
        img_cube = torch.ones((1,1, 1088, 192, 256))*-15.
        img_cube[:,:,:image.shape[2], :image.shape[3], :image.shape[4]] = image

        # nan -> -1
        img_cube[torch.isnan(img_cube)] = -15.
        
        # Normalization
        img_cube = img_cube.clamp(-15, 35)
        img_cube = (img_cube + 15) / 50

        return img_cube, origin_shape

    def postprocessing(self, pred, origin_shape):
        """
        params:
            pred: array [w,h,d]
            original_shape: tuple
        returns:
            pred_label: [w,h,d]
        """
        pred = pred>0.5
        # pred = morphology.closing(pred>0.5, morphology.cube(3))
        # pred = morphology.opening(pred, morphology.cube(3))
        label = measure.label(pred, connectivity=1)
        for prop in measure.regionprops(label):
            # print(prop.area, prop.bbox)
            if prop.area < 300 or ((prop.bbox[3] - prop.bbox[0])>2000) or ((prop.bbox[4] - prop.bbox[1])<2) or ((prop.bbox[5] - prop.bbox[2])<2):
                pred[label==prop.label] = 0
        pred_label = torch.nn.functional.interpolate(torch.as_tensor(pred[None, None]).float(), scale_factor=(4,1,1), mode='nearest')
        pred_label = pred_label[0, :, :origin_shape[0], :origin_shape[1], :origin_shape[2]]
        return pred_label


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    model_path = [
        './data/log/20230630-1508/weights/best_model_weights-0797-0.691.pkl',
        './data/log/20230817-1722/weights/best_model_weights-0737-0.713.pkl', 
        './data/log/20230824-0908/weights/best_model_weights-0725-0.702.pkl', 
        './data/log/20230808-1449/weights/best_model_weights-1149-0.716.pkl', 
        './data/log/20230913-0951/weights/best_model_weights-1122-0.721.pkl', 
                  ][-1]
    cube_path_list = sorted(glob.glob("/data/songzihao/data_ZY/cube/alfa-nii/Dec+2820_06_05__20220506_cube_Jy_*-image.nii.gz"))
    # cube_path_list = sorted(glob.glob("/data/songzihao/data_ZY/cube/MED300//fits/+*newbin3*fits"))
    # cube_path_list = sorted(glob.glob("/data/songzihao/data_ZY/cube/bootes-fits/*fits"))
    # cube_path_list = [x for x in cube_path_list if 'bin3_' in x]
    path_to_save = f"/data/songzihao/data_ZY/pred/{model_path.split('/')[3]}/{model_path.split('-')[-2]}"               ####
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    infer = Inference(model_path, path_to_save, 'UNetLK')            #####
    infer.predict(cube_path_list[:2])
