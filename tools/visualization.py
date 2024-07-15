import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from   tools.utils import convert_tensor_to_numpy
from   tools.utils import convert_nda_to_itk
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

def compute_jacobian_determinant(flow, i, j, k):
    du_dx = (flow[0, i+1, j, k] - flow[0, i-1, j, k]) / 2
    du_dy = (flow[0, i, j+1, k] - flow[0, i, j-1, k]) / 2
    du_dz = (flow[0, i, j, k+1] - flow[0, i, j, k-1]) / 2

    dv_dx = (flow[1, i+1, j, k] - flow[1, i-1, j, k]) / 2
    dv_dy = (flow[1, i, j+1, k] - flow[1, i, j-1, k]) / 2
    dv_dz = (flow[1, i, j, k+1] - flow[1, i, j, k-1]) / 2

    dw_dx = (flow[2, i+1, j, k] - flow[2, i-1, j, k]) / 2
    dw_dy = (flow[2, i, j+1, k] - flow[2, i, j-1, k]) / 2
    dw_dz = (flow[2, i, j, k+1] - flow[2, i, j, k-1]) / 2

    jacobian_matrix = np.array([
        [du_dx, du_dy, du_dz],
        [dv_dx, dv_dy, dv_dz],
        [dw_dx, dw_dy, dw_dz]
    ])
    
    return np.linalg.det(jacobian_matrix)
    

def get_jacobian_det(flow):
    '''
    flow: 3 x 192 x 192 x 208
    '''
    shape = flow.shape
    import pdb; pdb.set_trace
    jacob_det = np.zeros(shape[1:])
    
    for i in range(1, shape[1] - 1):
        for j in range(1, shape[2] - 1):
            for k in range(1, shape[3] - 1):
                jacob_det[i, j, k] = compute_jacobian_determinant(flow, i, j, k)
    return jacob_det

              
def save_heatmap_flow(flow, path_to_save):
    import os
    output_dir = path_to_save +'heatmap/'
    os.makedirs(output_dir, exist_ok=True) 
    flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
    
    for z in range(flow.shape[2]):
        plt.figure()
        plt.imshow(np.rot90(flow[:, :, z], 2), cmap='plasma')
        plt.colorbar(label='Deformation Magnitude')
        plt.title(f'Deformation Magnitude at Slice {z}')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'slice_{z:03d}.png'))
        plt.close()


def plot_deformation_field_with_grid(deformation_field, jacobian_determinant, path_to_save, cmap='plasma'):
    import os
    output_dir = path_to_save +'heatmap/' 
    os.makedirs(output_dir, exist_ok=True)

    # Normalize the determinant for visualization
    determinant_normalized = (jacobian_determinant - np.min(jacobian_determinant)) / (np.max(jacobian_determinant) - np.min(jacobian_determinant))
    
    # Normalize the deformation field for visualization
    deformation_magnitude = np.sqrt(np.sum(deformation_field**2, axis=0))
    #deformation_normalized = deformation_field / (deformation_magnitude.max() + 1e-8)
    
    for slice_index in range(deformation_field.shape[3]):
        # Rotate the slice by 180 degrees
        determinant_slice = np.rot90(determinant_normalized[:, :, slice_index], 2)
        
        # Plot the heatmap of the determinant
        plt.figure(figsize=(10, 10))
        plt.imshow(determinant_slice, cmap=cmap)
        plt.colorbar(label='Jacobian Determinant')
        
        # Overlay the deformation field as a grid
        step = 10  # Define the step size for the grid
        for i in range(0, deformation_field.shape[1], step):
            for j in range(0, deformation_field.shape[2], step):
                # Normalize the deformation vectors for each slice
                dx = deformation_field[0, i, j, slice_index] / (deformation_magnitude[i, j, slice_index] + 1e-8)
                dy = deformation_field[1, i, j, slice_index] / (deformation_magnitude[i, j, slice_index] + 1e-8)
                plt.quiver(j, i, dx, dy, color='black', alpha=0.5)
        
        plt.title(f'Jacobian Determinant and Deformation Field at Slice {slice_index}')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'slice_{slice_index:03d}_with_grid.png'))
        plt.close()


def plot_flow_with_arrows(image, flow, path_to_save):
    """
    Save the deformation field with arrows overlaid on each fixed image slice.

    Parameters:
    - image: 3D numpy array representing the fixed image.
    - flow: 4D numpy array with shape (3, 192, 192, 208), representing the deformation field.
                         The first dimension contains the displacement vectors (dy, dx, dz).
    - path_to_save: Directory to save the output images.
    """
    import os
    output_dir = path_to_save +'flow_arrows/' 
    os.makedirs(output_dir, exist_ok=True)
    
    for slice in range(image.shape[2]):
        image_slice = image[:, :, slice]
        flow_slice  = flow[:2, :, :, slice] # # Take only dy, dx for 2D slice
        
        h, w = image_slice.shape
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_slice, cmap='gray')
        plt.colorbar(label='Jacobian Determinant')
        plt.quiver(X, Y, flow_slice[1, :, :], flow_slice[0, :, :], color='r', angles='xy', scale_units='xy', scale=1)
        plt.title(f'Deformation Field with Arrows (Slice {slice})')
        plt.savefig(os.path.join(output_dir, f'deformation_field_arrows_slice_{slice}.png'))
        plt.close()



def save_outputs_as_nii_format(out, path_to_save='./output/'):
    itk_img1 = sitk.ReadImage(out['img1_p'])
    itk_img2 = sitk.ReadImage(out['img2_p'])
    img1  = np.squeeze(convert_tensor_to_numpy(out['img1']), axis=(0,1))
    img2  = np.squeeze(convert_tensor_to_numpy(out['img2']), axis=(0,1))   
    seg1  = np.squeeze(convert_tensor_to_numpy(out['seg1']), axis=(0,1))  
    seg2  = np.squeeze(convert_tensor_to_numpy(out['seg2']), axis=(0,1))  
    w_img = np.squeeze(convert_tensor_to_numpy(out['warped']), axis=(0,1)) 
    w_seg = np.squeeze(convert_tensor_to_numpy(out['wseg2']), axis=(0,1))  
    flow3 = np.squeeze(convert_tensor_to_numpy(out['flow']), axis=(0))  # 3 x 192 x 192 x 208
    flow  = np.linalg.norm(flow3, axis=0) # 192 x 192 x 208
    save_heatmap_flow(flow, path_to_save)
    #import pdb; pdb.set_trace()
    jdet  = get_jacobian_det(flow3)
    plot_deformation_field_with_grid(flow3, jdet, path_to_save)
    import pdb; pdb.set_trace()
    
    
    img1  = convert_nda_to_itk(img1, itk_img1)
    img2  = convert_nda_to_itk(img2, itk_img2)   
    seg1  = convert_nda_to_itk(seg1, itk_img1)
    seg2  = convert_nda_to_itk(seg2, itk_img2)  
    w_img = convert_nda_to_itk(w_img, itk_img1) 
    w_seg = convert_nda_to_itk(w_seg, itk_img1)  
    flow  = convert_nda_to_itk(flow, itk_img1)
    jdet  = convert_nda_to_itk(jdet, itk_img1) 
    import pdb; pdb.set_trace()
      
    
    sitk.WriteImage(img1, path_to_save + 'img1.nii.gz')
    sitk.WriteImage(img2, path_to_save + 'img2.nii.gz')
    sitk.WriteImage(seg1, path_to_save + 'seg1.nii.gz')
    sitk.WriteImage(seg2, path_to_save + 'seg2.nii.gz')
    sitk.WriteImage(w_img, path_to_save + 'w_img.nii.gz')
    sitk.WriteImage(w_seg, path_to_save + 'w_seg.nii.gz')
    sitk.WriteImage(flow, path_to_save + 'flow.nii.gz')
    sitk.WriteImage(jdet, path_to_save + 'jdet.nii.gz')