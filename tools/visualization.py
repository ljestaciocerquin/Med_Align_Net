import numpy as np
import matplotlib.pyplot as plt

def plot_sample_data(sample, slide=80, save_path=None):
    
    kps_i = np.zeros(sample['voxel1'][:, :, slide].shape)
    kps_e = np.zeros(sample['voxel2'][:, :, slide].shape)
    
    #for kp in sample['kps_i'][sample['kps_i'][:, 2] == slide]:
    #    kps_i[int(kp[1]), int(kp[0])] = 1
    #for kp in sample['kps_e'][sample['kps_e'][:, 2] == slide]:
    #    kps_e[int(kp[1]), int(kp[0])] = 1
    
    
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    axs[0,0].imshow(sample['voxel1'][:, :, slide], cmap='gray')
    axs[0,1].imshow(sample['voxel2'][:,:, slide], cmap='gray')
    axs[1,0].imshow(sample['segmentation1'][:,:, slide], cmap='gray')
    axs[1,1].imshow(sample['segmentation2'][:,:, slide], cmap='gray')
    #axs[2,0].imshow(kps_i, cmap='gray')
    #axs[2,1].imshow(kps_e, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    #axs[2, 0].axis('off')
    #axs[2, 1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()