import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from   natsort import natsorted
from   scipy.ndimage import gaussian_filter, map_coordinates

# Function to get the deformation field and moving images
def get_file_paths(directory):
    file_types = {
        'img1.nii.gz'   : [],
        'img2.nii.gz'   : [],
        'seg2.nii.gz'   : [],
        'w_img.nii.gz'  : [],
        'w_seg.nii.gz'  : [],
        'flow_4d.nii.gz': [],
        
    }
    
    for filename in os.listdir(directory):
        for file_suffix in file_types:
            if filename.endswith(file_suffix):
                file_types[file_suffix].append(os.path.join(directory, filename))
                break
    for key in file_types:
        file_types[key] = natsorted(file_types[key])
    
    return file_types


def load_nifti_image(file_path):
    nii   = nib.load(file_path)
    image = nii.get_fdata()
    return image, nii.affine

def load_deformation_field(file_path):
    nii = nib.load(file_path)
    deformation_field = nii.get_fdata()
    return deformation_field

def get_ct_views(ct_scan):
     # ct_scan: shape (192, 160, 256) => (z, y, x)
    
    # Axial view: Slices along the z-axis, output shape (160, 256) per slice
    axial_slices = [ct_scan[i, :, :] for i in range(ct_scan.shape[0])]
    
    # Sagittal view: Slices along the x-axis, output shape (192, 160) per slice
    sagittal_slices = [ct_scan[:, :, i] for i in range(ct_scan.shape[2])]
    
    # Coronal view: Slices along the y-axis, output shape (192, 256) per slice
    coronal_slices = [ct_scan[:, i, :] for i in range(ct_scan.shape[1])]
    
    return axial_slices, sagittal_slices, coronal_slices

def get_deformation_field_views(deformation_field):
    # deformation_field: shape (192, 160, 256, 3)
    
    # Axial view: Take slices along the z-axis (displacement in x and y directions)
    axial_deformation = [deformation_field[i, :, :, :2] for i in range(deformation_field.shape[0])]  # (160, 256, 2) per slice
    
    # Sagittal view: Take slices along the x-axis (displacement in y and z directions)
    sagittal_deformation = [deformation_field[:, :, i, 1:] for i in range(deformation_field.shape[2])]  # (192, 160, 2) per slice
    
    # Coronal view: Take slices along the y-axis (displacement in x and z directions)
    coronal_deformation = [deformation_field[:, i, :, ::2] for i in range(deformation_field.shape[1])]  # (192, 256, 2) per slice
    
    return axial_deformation, sagittal_deformation, coronal_deformation


def draw_grid(image, grid_spacing=10, grid_value=1):
    grid_image = np.zeros(image.shape)
    # Drawing the vertical and horizontal lines
    grid_image[:, ::grid_spacing] = grid_value
    grid_image[::grid_spacing, :] = grid_value
    return grid_image

def blur_grid(grid_image, sigma=0.25):
    blurred_grid = gaussian_filter(grid_image, sigma=sigma)
    return blurred_grid

def add_grid_to_image(image, grid_image):
    modified_image = image + grid_image
    return modified_image