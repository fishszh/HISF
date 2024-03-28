from networks.unet_lk import UNetLK
from networks.uxnet_3d import UXNET
from monai.networks.nets.swin_unetr import SwinUNETR
import os
import glob
import torch 
import torch.nn as nn
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
from astropy.io import fits
import torchio as tio
from monai.inferers import sliding_window_inference
from itertools import product

class Inference:
    def __init__(
            self, 
            model_path: str,
            path_to_save,
            model_type,
            patch_size
            ):
        self.init_model(model_path, model_type)
        self.path_to_save = path_to_save
        self.patch_size = patch_size

    def init_model(self, model_path, network_name):
        if network_name == 'UNetLK':
            model = UNetLK(1,1,32)
        elif network_name == 'UXNet':
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
        elif network_name == "swin-unetr":
            model = SwinUNETR(
                in_channels=1,
                img_size=(1024,32,64),
                out_channels=1,
        )

        model.eval()
        model_weights = torch.load(model_path)
        model.load_state_dict(model_weights)
        self.model = nn.DataParallel(model).cuda()

    def predict(self, cube_path_list):
        for cube_path in tqdm(cube_path_list):
            label = self.predict_single_cube(cube_path)



    def predict_single_cube(self, cube_path):
        patches, centers, origin_shape, target_shape = self.preprocessing(cube_path)
        prediction = torch.zeros(target_shape)
        predictionx = torch.zeros(target_shape)
        with torch.no_grad():
            for i, patch in enumerate(patches):
                patch = patch.cuda().float().unsqueeze(dim=0)
                pred = self.model(patch).sigmoid().detach().cpu()[0,0]
                prediction[
                    centers[i][0] - self.patch_size[0]//2 : centers[i][0] + self.patch_size[0]//2,
                    centers[i][1] - self.patch_size[1]//2 : centers[i][1] + self.patch_size[1]//2,
                    centers[i][2] - self.patch_size[2]//2 : centers[i][2] + self.patch_size[2]//2,
                ] += pred
                predictionx[
                    centers[i][0] - self.patch_size[0]//2 : centers[i][0] + self.patch_size[0]//2,
                    centers[i][1] - self.patch_size[1]//2 : centers[i][1] + self.patch_size[1]//2,
                    centers[i][2] - self.patch_size[2]//2 : centers[i][2] + self.patch_size[2]//2,
                ] += 1
        prediction /= predictionx

        label = self.postprocessing(prediction)
        
        label = tio.LabelMap(tensor=torch.tensor(label).int().unsqueeze(dim=0))
        label = tio.Resize(origin_shape)(label)
        label_path = f"{self.path_to_save}/{cube_path.split('/')[-1]}"
        # label = tio.ScalarImage(tensor=torch.as_tensor(label[None]))
        label.affine = np.array([
            [1.0,0,0,0],
            [0,1.0,0,0],
            [0,0,1.0,0],
            [0,0,0,1.0],
        ])
        label.save(label_path.replace('image', 'pred-label'))
        return label

    def preprocessing(self, cube_path):
        image = tio.ScalarImage(cube_path)
        origin_shape = image.shape[1:]
        target_shape = (1024,32,256)
        img = torch.nn.functional.avg_pool3d(image.data[None], kernel_size=(6,1,1), stride=(4,1,1))
        image.set_data(img[0])
        image = tio.Resize(target_shape)(image)
        image = image.data
        image = image.clamp(-15, 35)
        image = (image + 15) / 50

        centers = self._get_centers(image.shape[1:], self.patch_size)
        patches = [
            image[:,
                  cent[0] - self.patch_size[0]//2 : cent[0] + self.patch_size[0]//2,
                  cent[1] - self.patch_size[1]//2 : cent[1] + self.patch_size[1]//2,
                  cent[2] - self.patch_size[2]//2 : cent[2] + self.patch_size[2]//2
                  ]
            for cent in centers
        ]
        return patches, centers, origin_shape, target_shape

    def _get_centers(self, image_shape, patch_size):
        dim_coords = [
            list(range(0, image_shape[i], patch_size[i]//2))[1:-1] + [image_shape[i] - patch_size[i]//2] 
            for i in range(len(image_shape))]
        return list(product(*dim_coords))
    
    def postprocessing(self, pred):
        """
        params:
            pred: array [w,h,d]
            original_shape: tuple
        returns:
            pred_label: [w,h,d]
        """
        pred = pred > 0.5
        pred = morphology.closing(pred>0.5, morphology.cube(3))
        pred = morphology.opening(pred, morphology.cube(3))
        label = measure.label(pred, connectivity=1)
        for prop in measure.regionprops(label):
            # print(prop.area, prop.bbox)
            if prop.area < 300 or ((prop.bbox[3] - prop.bbox[0])>2000) or ((prop.bbox[4] - prop.bbox[1])<2) or ((prop.bbox[5] - prop.bbox[2])<2):
                pred[label==prop.label] = 0
        return pred


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    model_path = [
        './data/log/20240313-1800/weights/best_model_weights-0850-0.420.pkl',
        './data/log/20240314-1501/weights/best_model_weights-0786-0.464.pkl',
        './data/log/20240321-1033/weights/best_model_weights-0657-0.517.pkl',

                  ][-1]
    # cube_path_list = sorted(glob.glob("/data/songzihao/data_ZY/cube/alfa-nii/Dec+2820_06_05__20220506_cube_Jy_*-image.nii.gz"))
    cube_path_list = sorted(glob.glob("/data/songzihao/data_ZY/cube/alfa-nii/*-image.nii.gz"))
    path_to_save = f"/data/songzihao/data_ZY/pred/{model_path.split('/')[3]}/{model_path.split('-')[-2]}"               ####
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    infer = Inference(model_path, path_to_save, 'swin-unetr', patch_size=(1024,32,64))
    infer.predict(cube_path_list[::7])
