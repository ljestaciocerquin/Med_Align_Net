import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import SimpleITK as sitk


def convert_tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor

def load_model(state_dict, model):
    # load state dict
    model.stems.load_state_dict(state_dict['stem_state_dict'])
    # if hypernet in model attribute
    if hasattr(model, 'hypernet'):
        model.hypernet.load_state_dict(state_dict['hypernet_state_dict'])
    return model


def load_model_from_dir(checkpoint_dir, model):
    # glob file with suffix pth
    from pathlib import Path as pa
    import re
    p = pa(checkpoint_dir)
    # check if p has subdir named model_wts
    if (p/'model_wts').exists():
        p = p/'model_wts'
    p = p.glob('*.pth')
    p = sorted(p, key=lambda x: [int(n) for n in re.findall(r'\d+', str(x))])
    model_path = str(p[-1])
    load_model(torch.load(model_path), model)
    return model_path


def find_surf(seg, kernel=3, thres=1):
    '''
    Find near-surface voxels of a segmentation.

    Args:
        seg: (**,D,H,W)
        radius: int

    Returns:
        surf: (**,D,H,W)
    '''
    if thres<=0:
        return torch.zeros_like(seg).bool()
    pads    = tuple((kernel-1)//2 for _ in range(6))
    seg_k   = F.pad(seg, pads, mode='constant', value=0).unfold(-3, kernel, 1).unfold(-3, kernel, 1).unfold(-3, kernel, 1)
    seg_num = seg_k.sum(dim=(-1,-2,-3))
    surf    = (seg_num<(kernel**3)*thres) & seg.bool()
    # how large a boundary we want to remove?
    # surf = (seg_num<(kernel**3//2)) & seg.bool()
    return surf


def show_img(res, save_path=None, norm=True, cmap=None, inter_dst=5) -> Image:
    import torchvision.transforms as T
    res = tt(res)
    if norm: res = normalize(res)
    if res.ndim>=3:
        return T.ToPILImage()(visualize_3d(res, cmap=cmap, inter_dst=inter_dst))
    # normalize res
    # res = (res-res.min())/(res.max()-res.min())

    pimg = T.ToPILImage()(res)
    if save_path:
        pimg.save(save_path)
    return pimg


def convert_nda_to_itk(nda: np.ndarray, itk_image: sitk.Image):
    """From a numpy array, get an itk image object, copying information
    from an existing one. It switches the z-axis from last to first position.

    Args:
        nda (np.ndarray): 3D image array
        itk_image (sitk.Image): Image object to copy info from

    Returns:
        new_itk_image (sitk.Image): New Image object
    """
    new_itk_image = sitk.GetImageFromArray(np.moveaxis(nda, -1, 0))
    new_itk_image.SetOrigin(itk_image.GetOrigin())
    new_itk_image.SetSpacing(itk_image.GetSpacing())
    new_itk_image.CopyInformation(itk_image)
    return new_itk_image

def convert_itk_to_nda(itk_image: sitk.Image):
    """From an itk Image object, get a numpy array. It moves the first z-axis
    to the last position (np.ndarray convention).

    Args:
        itk_image (sitk.Image): Image object to convert

    Returns:
        result (np.ndarray): Converted nda image
    """
    return np.moveaxis(sitk.GetArrayFromImage(itk_image), 0, -1)

