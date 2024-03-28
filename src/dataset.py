import torch
from skimage import measure
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchio as tio
import glob
import random

class HICropDataSet(Dataset):
    def __init__(self, batch_size, num_samples=4, crop_size=(1024,32,64), train=True):
        # self.cdms_path = cdms_path
        self.batch_size = batch_size
        self.num_samples = num_samples # 
        self.crop_size = crop_size
        self.train = train
        self._data_list()

    def __len__(self):
        if self.train:
            return len(self.train_paired_path)
        else:
            return len(self.valid_paired_path)
    
    def _data_list(self):
        labell = sorted(glob.glob(f"/data/songzihao/data_ZY/cube/manual-annotation-eval/*label.nii.gz"))
        imagel = [x.replace('manual-annotation-eval', 'alfa-nii').replace('label', 'image') for x in labell]
        paired_path = list(zip(imagel, labell))
        self.valid_paired_path = paired_path[::7]
        self.train_paired_path = [x for x in paired_path if x not in self.valid_paired_path]
        

    def __getitem__(self, idx):
        # background
        std = 2.8 + torch.rand(1)
        background = torch.randn((1,) + self.crop_size)
        background = background * std / background.std()
        mask = torch.zeros_like(background)

        if self.train:
            idx = np.random.randint(0, len(self.train_paired_path))
            image = tio.ScalarImage(self.train_paired_path[idx][0])
            label = tio.LabelMap(self.train_paired_path[idx][1])
        else:
            idx = np.random.randint(0, len(self.valid_paired_path))
            image = tio.ScalarImage(self.valid_paired_path[idx][0])
            label = tio.LabelMap(self.valid_paired_path[idx][1])
        # image = tio.Resize(target_shape=(image.shape[-3],32,image.shape[-1]))(image)
        # label = tio.Resize(target_shape=(image.shape[-3],32,image.shape[-1]), label_interpolation='linear')(label)
        img = torch.nn.functional.avg_pool3d(image.data[None], kernel_size=(6,1,1), stride=(4,1,1))
        lbl = torch.nn.functional.avg_pool3d(label.data[None].float(), kernel_size=(6,1,1), stride=(4,1,1))
        lbl = (lbl>0.5).float()
        image.set_data(img[0])
        label.set_data(lbl[0])
        image = tio.Resize(target_shape=(1024,32,256))(image)
        label = tio.Resize(target_shape=(1024,32,256), label_interpolation='linear')(label)
        label.set_data((label.data>0.5).float())

        # calculate rois bbox
        roi_bboxs = self._get_roi_bbox(label.data[0])
        roi_crop_bboxs = [self._get_roi_crop_bbox(label.data[0].shape, bbox) for bbox in roi_bboxs]

        # crop rois
        image_rois = [self._crop_roi(image.data[0], bbox) for bbox in roi_crop_bboxs]
        label_rois = [self._crop_roi(label.data[0], bbox) for bbox in roi_crop_bboxs]

        # Normalization
        image_rois = torch.stack(image_rois).unsqueeze(dim=1).float()
        label_rois = torch.stack(label_rois).unsqueeze(dim=1).float()
        image_rois = image_rois.clamp(-15,35)
        image_rois = (image_rois + 15) / 50

        return image_rois, label_rois
    
    def _get_roi_bbox(self,  label_arr):
        # generate positvie samples' bbox
        pos_bboxs = self._get_pos_bbox(label_arr)

        # generate negative samples' bbox
        neg_bboxs = self._get_neg_bbox(label_arr.shape)

        roi_bboxs = random.sample(pos_bboxs+neg_bboxs, k=self.num_samples)
        return roi_bboxs
    
    def _get_roi_crop_bbox(self, image_shape, bbox):
        centorid = [(bbox[i]+bbox[i+3])//2 for i in range(3)]
        # freq random start index
        #  0 <= start <= center
        c1 = np.arange(centorid[0]+1)
        #  c <= start + self.crop_size[0] <= arr.shape[0]
        c2 = np.arange(centorid[0], image_shape[0]+1) - self.crop_size[0]
        xi = np.random.choice(np.intersect1d(c1,c2))
        
        c1 = np.arange(centorid[2]+1)
        c2 = np.arange(centorid[2], image_shape[2]+1) - self.crop_size[2]
        zi = np.random.choice(np.intersect1d(c1,c2))

        return [xi, 0, zi, xi+self.crop_size[0], self.crop_size[1], zi+self.crop_size[2]]

    
    def _crop_roi(self, arr, bbox):
        roi = arr[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        return roi

    def _get_pos_bbox(self, label_arr):
        bboxs = [prop.bbox for prop in measure.regionprops(measure.label(label_arr))]
        return bboxs

    def _get_neg_bbox(self, image_shape):
        # randomly crop
        xs = np.random.randint(20,image_shape[0]-20, self.num_samples)
        ys = np.random.randint(5,image_shape[1]-5, self.num_samples)
        zs = np.random.randint(5,image_shape[2]-5, self.num_samples)
        xe = xs + 10
        ye = ys + 2
        ze = zs + 2
        neg_bboxs = list(zip(xs, ys, zs, xe, ye, ze))
        return neg_bboxs
        

    @staticmethod
    def collate_fn(samples):
        images = torch.concat([x[0] for x in samples], dim=0)
        labels = torch.concat([x[1] for x in samples], dim=0)
        return images, labels

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle, 
            num_workers=num_workers, collate_fn=HICropDataSet.collate_fn)


class HICubeDataSet(Dataset):
    def __init__(self, batch_size, num_samples, target_shape=(1024,32,256), train=True):
        # self.cdms_path = cdms_path
        # self.num_samples = num_samples
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.train = train
        self._data_list()

    def __len__(self):
        if self.train:
            return len(self.train_paired_path)
        else:
            return len(self.valid_paired_path)

    def _data_list(self):
        labell = sorted(glob.glob(f"/data/songzihao/data_ZY/cube/manual-annotation-eval/*label.nii.gz"))
        imagel = [x.replace('manual-annotation-eval', 'alfa-nii').replace('label', 'image') for x in labell]
        paired_path = list(zip(imagel, labell))
        self.valid_paired_path = paired_path[::7]
        self.train_paired_path = [x for x in paired_path if x not in self.valid_paired_path]
        crop_lbpl = sorted(glob.glob(f"/data/songzihao/data_ZY/cube/manual-annotation-eval-crop/*label.nii.gz"))
        crop_impl = sorted(glob.glob(f"/data/songzihao/data_ZY/cube/manual-annotation-eval-crop/*image.nii.gz"))
        self.crop_lbpl = [x for x in crop_lbpl if '-3-label' not in x]
        self.crop_impl = [x for x in crop_impl if '-3-image' not in x]

        

    def __getitem__(self, idx):
        # background
        std = 2.8 + torch.rand(1)
        noise = torch.randn((1,) + self.target_shape)
        noise = noise * std / noise.std()

        if self.train:
            idx = np.random.randint(0, len(self.train_paired_path))
            image = tio.ScalarImage(self.train_paired_path[idx][0])
            label = tio.LabelMap(self.train_paired_path[idx][1])
        else:
            idx = np.random.randint(0, len(self.valid_paired_path))
            image = tio.ScalarImage(self.valid_paired_path[idx][0])
            label = tio.LabelMap(self.valid_paired_path[idx][1])
        img = torch.nn.functional.avg_pool3d(image.data[None], kernel_size=(6,1,1), stride=(4,1,1))
        lbl = torch.nn.functional.avg_pool3d(label.data[None].float(), kernel_size=(6,1,1), stride=(4,1,1))
        lbl = (lbl>0.5).float()
        image.set_data(img[0])
        label.set_data(lbl[0])
        image = tio.Resize(target_shape=self.target_shape)(image)
        label = tio.Resize(target_shape=self.target_shape, label_interpolation='linear')(label)
        # mix noise
        image = image.data + noise
        label = (label.data>0.5).float()

        # random foreground mix up 
        if np.random.uniform(0,1) > 0.6:
            num_fore = np.random.randint(1, 4)
            for _ in range(num_fore):
                i = np.random.randint(0, len(self.crop_lbpl))
                fimage = tio.ScalarImage(self.crop_impl[i])
                flabel = tio.LabelMap(self.crop_lbpl[i])

                percent = np.random.randint(92,98)
                intensity_thresh = np.percentile(fimage.data, percent)
                fimage.set_data(fimage.data*std/intensity_thresh)
                
                # fimage = tio.Resize(target_shape=(fimage.shape[-3]//4, round(fimage.shape[-2]*1.4), fimage.shape[-1]))(fimage)
                # flabel = tio.Resize(target_shape=(flabel.shape[-3]//4, round(flabel.shape[-2]*1.4), flabel.shape[-1]), label_interpolation='linear')(flabel)
                fimage = torch.nn.functional.avg_pool3d(fimage.data[None], kernel_size=(6,1,1), stride=(4,1,1))
                flabel = torch.nn.functional.avg_pool3d(flabel.data[None].float(), kernel_size=(6,1,1), stride=(4,1,1))
                flabel = (flabel>0.5).float()
                # fimage = fimage.data

                ix = np.random.randint(0, self.target_shape[0]-fimage.shape[-3])
                iy = np.random.randint(0, self.target_shape[1]-fimage.shape[-2])
                iz = np.random.randint(0, self.target_shape[2]-fimage.shape[-1])
                image[:, ix:ix+fimage.shape[-3], iy:iy+fimage.shape[-2], iz:iz+fimage.shape[-1]] += fimage[0]
                label[:, ix:ix+fimage.shape[-3], iy:iy+fimage.shape[-2], iz:iz+fimage.shape[-1]] += flabel[0]
            label = (label>0.5).float() 
        
        # Augmentation
        if np.random.uniform(0,1) > 0.5:
            image = torch.flip(image, [1])
            label = torch.flip(label, [1])
        if np.random.uniform(0,1) > 0.5:
            image = torch.flip(image, [2])
            label = torch.flip(label, [2])
        if np.random.uniform(0,1) > 0.5:
            image = torch.flip(image, [3])
            label = torch.flip(label, [3])
        
        # Normalization
        image = image.clamp(-15,35)
        image = (image + 15) / 50
        return image, label

    @staticmethod
    def collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        labels = torch.stack([x[1] for x in samples])
        return images, labels

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle, 
            num_workers=num_workers, collate_fn=HICubeDataSet.collate_fn)




if __name__ == '__main__':
    from tqdm import tqdm
    hids = HICubeDataSet(2, 4)
    # hidl = HIDataSet.get_dataloader(hids, batch_size=4)

    for i in tqdm(range(5)):
        image, label = hids[i]
        # image = tio.ScalarImage(tensor=image[0])
        # label = tio.LabelMap(tensor=label[0])
        # print(image.shape, label.shape)
        # image.save(f'./{i}-image.nii.gz')
        # label.save(f'./{i}-label.nii.gz')
    # for images, labels in hidl:
    #     print(images)



                