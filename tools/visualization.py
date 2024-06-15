import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from   tools.utils import convert_tensor_to_numpy
from torchvision.utils import make_grid, save_image, draw_segmentation_masks


def plot_sample_data(sample, slide=80, save_path=None):
    
    voxel1 = convert_tensor_to_numpy(sample['voxel1'])
    voxel2 = convert_tensor_to_numpy(sample['voxel2'])
    segmentation1 = convert_tensor_to_numpy(sample['segmentation1'])
    segmentation2 = convert_tensor_to_numpy(sample['segmentation2'])
    kps_i = np.zeros(voxel1[:, :, slide].shape)
    kps_e = np.zeros(voxel2[:, :, slide].shape)
    
    # Uncomment if you need to visualize keypoints
    kps_i_coords = convert_tensor_to_numpy(sample['kps1'])
    kps_e_coords = convert_tensor_to_numpy(sample['kps2'])
    for kp in kps_i_coords[kps_i_coords[:, 2] == slide]:
        kps_i[int(kp[1]), int(kp[0])] = 1
    for kp in kps_e_coords[kps_e_coords[:, 2] == slide]:
        kps_e[int(kp[1]), int(kp[0])] = 1
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].imshow(voxel1[:, :, slide], cmap='gray') # slide shouls be the last axis
    axs[0, 1].imshow(voxel2[:, :, slide], cmap='gray')
    axs[1, 0].imshow(segmentation1[:, :, slide], cmap='gray')
    axs[1, 1].imshow(segmentation2[:, :, slide], cmap='gray')
    # Uncomment if you need to visualize keypoints
    axs[2, 0].imshow(kps_i, cmap='gray')
    axs[2, 1].imshow(kps_e, cmap='gray')
    
    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def print_img_info(selected_image, title='Train image:'):
    print(title)
    print('origin: ' + str(selected_image.GetOrigin()))
    print('size: ' + str(selected_image.GetSize()))
    print('spacing: ' + str(selected_image.GetSpacing()))
    print('direction: ' + str(selected_image.GetDirection()))
    print('pixel type: ' + str(selected_image.GetPixelIDTypeAsString()))
    print('number of pixel components: ' + str(selected_image.GetNumberOfComponentsPerPixel()))


# a simple function to plot an image
def plot1(fixed, title='', slice=128, figsize=(12, 12)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.imshow(sitk.GetArrayFromImage(fixed)[slice, :, :], cmap='gray', origin='lower')
    axs.set_title(title, fontdict={'size':26})
    axs.axis('off')
    plt.tight_layout()
    plt.show()
    
# a simple function to plot 3 images at once
def plot3(fixed, moving, transformed, labels=['Fixed', 'Moving', 'Moving Transformed'], slice=128):
    fig, axs = plt.subplots(1, 3, figsize=(24, 12))
    axs[0].imshow(sitk.GetArrayFromImage(fixed)[slice, :, :], cmap='gray', origin='lower')
    axs[0].set_title(labels[0], fontdict={'size':26})
    axs[0].axis('off')
    axs[1].imshow(sitk.GetArrayFromImage(moving)[slice, :, :], cmap='gray', origin='lower')
    axs[1].axis('off')
    axs[1].set_title(labels[1], fontdict={'size':26})
    axs[2].imshow(sitk.GetArrayFromImage(transformed)[slice, :, :], cmap='gray', origin='lower')
    axs[2].axis('off')
    axs[2].set_title(labels[2], fontdict={'size':26})
    plt.tight_layout()
    plt.show()
    
    
    
def normalize(data, dim=3, ct=False):
    data = tt(data)
    dim = min(3, data.dim())
    if ct:
        data1 = data.flatten(start_dim=-dim)
        l = data1.shape[-1]
        upper = data.kthvalue(int(0.95*l), dim=-1)
        lower = data.kthvalue(int(0.05*l), dim=-1)
        data = data.clip(lower, upper)
    return PyTMinMaxScalerVectorized()(data, dim=dim)

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor: torch.Tensor, dim=2):
        """
        tensor: N*C*H*W"""
        tensor = tensor.clone()
        s = tensor.shape
        tensor = tensor.flatten(-dim)
        scale = 1.0 / (
            tensor.max(dim=-1, keepdim=True)[0] - tensor.min(dim=-1, keepdim=True)[0]
        )
        tensor.mul_(scale).sub_(tensor.min(dim=-1, keepdim=True)[0])
        return tensor.view(*s)
    
    
tt = torch.as_tensor    
def visualize_3d(data, width=5, inter_dst=5, save_name=None, print_=False, color_channel: int=None, norm: bool=False, cmap=None):
    """
    data: (S, H, W) or (N, C, H, W)"""
    data =tt(data)
    if norm:
        data = normalize(data)
    img = data.float()
    # st = torch.tensor([76, 212, 226])
    # end = st+128
    # img = img[st[0]:end[0],st[1]:end[1],st[2]:end[2]]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.dim() < 4:
        img = img[:, None]
    img_s = img[::inter_dst]
    if color_channel is not None:
        img_s = img_s.movedim(color_channel, 1)

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    if cmap:
        alpha = img_s.cpu()
        img_s = cmap(img_s.cpu().numpy()*255) # cmap range is from 0-255
        img_s = torch.from_numpy(img_s).float().squeeze().permute(0, 3, 1, 2)
        img_s[:,[-1]] = alpha # add the alpha channel
    img_f = make_grid(img_s, nrow=width, padding=5, pad_value=1, normalize=True) # make_grid assumes that the input is in range [0, 1]
    if save_name:
        save_image(img_f, save_name)
        if print_:
            print("Visualizing img with shape and type:", img_s.shape, img_s.dtype, "on path {}".format(save_name) if save_name else "")
        return range(0, img.shape[0], inter_dst)
    else:
        return img_f
    
def draw_seg_on_vol(data, lb, if_norm=True, alpha=0.3, colors=["green", "red", "blue"], to_onehot=False, inter_dst=1):
    """
    Plot a 3D volume with binary segmentation masks overlaid on it.

    Parameters:
        data (torch.Tensor): The input 3D volume, shape: ((1,) S, H, W).
        lb (torch.Tensor): Binary masks representing segmentations, shape: ((M,) S, H, W).
        if_norm (bool): Whether to normalize the input volume. Default is True.
        alpha (float): Transparency of the overlay masks. Default is 0.3.
        colors (list): List of colors to use for overlay masks. Default is ["green", "red", "blue"].
        to_onehot (bool): Whether to convert the input masks to one-hot encoding. Default is False.

    Returns:
        torch.Tensor: Normalized output volume with overlay masks, shape: (S, 3, H, W).
    """
    data = data[...,::inter_dst,:,:]
    lb = lb[...,::inter_dst,:,:]
    if to_onehot:
        # check lb type long
        assert lb.dtype == torch.long or np.issubdtype(lb.dtype, np.integer), "lb should be integer"
        # remove class 0 (assume background)
        lb = F.one_hot(lb).moveaxis(3,0)[1:]
    lb = tt(lb).reshape(-1, *lb.shape[-3:])
    data =tt(data).float().reshape(1, *data.shape[-3:])
    if if_norm:
        data = normalize(data, 3)
    data = (data*255).cpu().to(torch.uint8)
    res = []
    for d, l in zip(data.transpose(0,1), lb.cpu().transpose(0,1)):
        res.append(draw_segmentation_masks(
                            (d).repeat(3, 1, 1),
                            l.bool(),
                            alpha=alpha,
                            colors=colors,
                        ))
    return torch.stack(res)/255

