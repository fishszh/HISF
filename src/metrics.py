from skimage import measure
import numpy as np

class Metrics:

    def dice(x, y):
        x = x.sigmoid()
        y = (y>0.3).float()
        i, u = [t.flatten(1).sum(1) for t in [x * y, x + y]]
        dc = ((2 * i + 1) / (u + 1)).mean()
        return dc

    def counts(x, y):
        x = x.sigmoid().detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        TP, FN, FP = 0, 0, 0
        for ii in range(x.shape[0]):
            label_gt = measure.label(y[ii,0], connectivity=2)
            label_pred = measure.label(x[ii,0]>0.5, connectivity=2)
            for prop in measure.regionprops(label_pred):
                if prop.area < 300:
                    x[ii,0][label_pred==prop.label] = 0
            label_pred = measure.label(x[ii,0]>0.5, connectivity=2)

            props_gt = measure.regionprops(label_gt)
            props_pred = measure.regionprops(label_pred)

            labelid_gt = list(range(1, len(props_gt)+1))
            labelid_pred = list(range(1, len(props_pred)+1))
            paired_label = []
            for i in labelid_gt:
                for j in labelid_pred:
                    a,b = (label_gt==i).astype(np.int8), (label_pred==j).astype(np.int8)
                    dice = 2*(a*b+1).sum()/(a+b+1).sum()
                    if dice > 0.3:
                        paired_label.append((i,j))
                        labelid_pred.remove(j)
                        break
            TP += len(paired_label)
            FN += len(props_gt) - TP
            FP += len(props_pred) - TP
        return TP, FN, FP


