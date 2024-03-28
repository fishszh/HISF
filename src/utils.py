import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import measure
import numpy as np

def demo_predict_cube_mask(feature, label, prediction, mode, epoch, step_train, path_demo):

    fore = feature.copy()
    pred = feature.copy()

    fore[label<0.5] = 0
    pred[prediction<0.5] = 0

    fig = plt.figure(figsize=(20, 15), layout="constrained")
    gs = GridSpec(3, 20, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:19])
    ax2 = fig.add_subplot(gs[0, 19:])
    ax3 = fig.add_subplot(gs[1, 0:19])
    ax4 = fig.add_subplot(gs[1, 19:])
    ax5 = fig.add_subplot(gs[2, :19])

    ax1.imshow(fore.sum(1).T, cmap='Blues', aspect='auto')
    ax2.imshow(fore.sum(0).T, cmap='Blues', aspect='auto')
    ax3.imshow(pred.sum(1).T, cmap='Oranges', aspect='auto')
    ax4.imshow(pred.sum(0).T, cmap='Oranges', aspect='auto')
    ax5.plot(fore.sum((1,2)), c='b', label='gt')
    ax5.plot(pred.sum((1,2)), c='orange', label='pred')
    plt.legend()
    ax5.set_xlim(0,fore.shape[0])
    ax1.grid(linestyle=':')
    ax2.grid(linestyle=':')
    ax3.grid(linestyle=':')
    ax4.grid(linestyle=':')
    plt.savefig(f"{path_demo}/{mode}_{epoch:04d}_{step_train:04d}_mask.png")
    plt.close()