import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from   natsort import natsorted

# Function to get the Jacobian determinat files
def get_file_paths(directory):
    file_types = {
        'jdet.nii.gz': [],
        'img1.nii.gz': [],
        'img2.nii.gz': [],
        'seg1.nii.gz': [],
        'seg2.nii.gz': [],
        'w_img.nii.gz': [],
        'w_seg.nii.gz': [],
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

# Function to visualize a 3D image with consistent color scale
def visualize_jacobiannnnn(jacobian_data, name, color_range=(0, 2), cmap='jet', ):
    # Plot slices from the 3D data
    fig, axes = plt.subplots(1, figsize=(15, 5))
    
    # Select slices (middle slice in each dimension)
    slice_x =  np.transpose(jacobian_data[jacobian_data.shape[0] // 3, :, :], (1, 0))
    slice_y =  np.transpose(jacobian_data[:, jacobian_data.shape[1] // 3, :], (1, 0))
    #slice_z =  np.transpose(jacobian_data[:, :, jacobian_data.shape[2] // 3], (1, 0))
    slice_z =  np.transpose(jacobian_data[:, :, 157], (1, 0))

    #slices = [slice_x, slice_y, slice_z]
    #slices = [slice_z, slice_z]
    #titles = ['Sagittal Slice', 'Coronal Slice', 'Axial Slice']

    '''for i, ax in enumerate(axes):
        im = ax.imshow(slices[i], cmap=cmap, origin='lower',
                       vmin=color_range[0], vmax=color_range[1])
        #ax.set_title(titles[i])
        ax.axis('off')
        #fig.colorbar(im, ax=ax)'''
    im = ax.imshow(slice_z, cmap=cmap, origin='lower',
                       vmin=color_range[0], vmax=color_range[1])
    fig.colorbar(im, ax=ax)
    plt.savefig(name)
    plt.close()

def visualize_jacobian(jacobian_data, filename, idx, color_range=(0, 2), cmap='jet'):
    """
    Visualizes a slice from 3D Jacobian data and saves the plot as an image.

    Parameters:
    - jacobian_data: 3D numpy array containing the Jacobian data.
    - filename: Name of the file to save the plot.
    - color_range: Tuple specifying the color scale range (min, max).
    - colormap: Colormap to use for visualization.
    """
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # Select a specific slice for visualization (e.g., z-slice)
    slice_index = idx  # Adjust this index as needed based on data dimensions
    slice_z = np.transpose(jacobian_data[:, :, slice_index], (1, 0))

    # Display the slice using imshow
    image = ax.imshow(slice_z, cmap=cmap, origin='lower',
                      vmin=color_range[0], vmax=color_range[1])

    # Add a colorbar for reference
    
    #cbar = fig.colorbar(image, ax=ax)
    #cbar.set_label('Value')  # Label for the colorbar

    # Set the title and labels
    #ax.set_title(f'Jacobian Slice at Index {slice_index}')
    ax.axis('off')  # Hide the axes for a cleaner look

    # Save the figure
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free up memory


def visualize_slice_img(img1, filename, idx, cmap='gray'):
    """
    Visualizes a slice from 3D Jacobian data and saves the plot as an image.

    Parameters:
    - jacobian_data: 3D numpy array containing the Jacobian data.
    - filename: Name of the file to save the plot.
    - color_range: Tuple specifying the color scale range (min, max).
    - colormap: Colormap to use for visualization.
    """
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # Select a specific slice for visualization (e.g., z-slice)
    slice_index = idx  # Adjust this index as needed based on data dimensions
    slice_img1 = np.transpose(img1[:, :, slice_index], (1, 0))

    # Display the slice using imshow
    image = ax.imshow(slice_img1, cmap=cmap)
    ax.axis('off')
    # Save the figure
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free up memory


# VoxelMorph Directory
vxm_directory = '/data/groups/beets-tan/l.estacio/Med_Align_Net/Jun24-204642_lutrain_VXMx1___/'
jdet_dir = './jdet_figures_lung/'
image_paths   = get_file_paths(vxm_directory)

jdet_paths    = image_paths['jdet.nii.gz']
jdet_0_path = jdet_paths[0] # 31_34 worst
jdet_1_path = jdet_paths[1] # 35_40 better
print('len: ', len(jdet_paths))
color_range = (0, 2)
idx0 = 104
idx1 = 104

img, _ = load_nifti_image(jdet_0_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'vxm_0.png', idx0, color_range=color_range, cmap='seismic')
visualize_slice_img(img, jdet_dir + 'vxm_0_det.png', idx0)

color_range        = (0, 2)
img, _ = load_nifti_image(jdet_1_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'vxm_1.png', idx1, color_range=color_range, cmap='seismic')
visualize_slice_img(img, jdet_dir + 'vxm_1_det.png', idx1)


# Imgs
img0_paths  = image_paths['img1.nii.gz']
fix_0_path = img0_paths[0]
fix_1_path = img0_paths[1]
img1_paths = image_paths['img2.nii.gz']
mov_0_path = img1_paths[0]
mov_1_path = img1_paths[1]

seg0_paths  = image_paths['seg1.nii.gz']
fix_seg_0_path = seg0_paths[0]
fix_seg_1_path = seg0_paths[1]
seg1_paths = image_paths['seg2.nii.gz']
mov_seg_0_path = seg1_paths[0]
mov_seg_1_path = seg1_paths[1]

w_img_paths  = image_paths['w_img.nii.gz']
w_img_0_path = w_img_paths[0]
w_img_1_path = w_img_paths[1]
w_seg_paths  = image_paths['w_seg.nii.gz']
w_seg_0_path = w_seg_paths[0]
w_seg_1_path = w_seg_paths[1]


fix_0, _ = load_nifti_image(fix_0_path)
fix_1, _ = load_nifti_image(fix_1_path)
mov_0, _ = load_nifti_image(mov_0_path)
mov_1, _ = load_nifti_image(mov_1_path)

fix_seg_0, _ = load_nifti_image(fix_seg_0_path)
fix_seg_1, _ = load_nifti_image(fix_seg_1_path)
mov_seg_0, _ = load_nifti_image(mov_seg_0_path)
mov_seg_1, _ = load_nifti_image(mov_seg_1_path)

w_img_0, _ = load_nifti_image(w_img_0_path)
w_img_1, _ = load_nifti_image(w_img_1_path)
w_seg_0, _ = load_nifti_image(w_seg_0_path)
w_seg_1, _ = load_nifti_image(w_seg_0_path)

visualize_slice_img(fix_0, jdet_dir + 'fix_0.png', idx0)
visualize_slice_img(fix_seg_0, jdet_dir + 'fix_seg_0.png', idx0)
visualize_slice_img(mov_0, jdet_dir + 'mov_0.png', idx0)
visualize_slice_img(mov_seg_0, jdet_dir + 'mov_seg_0.png', idx0)

visualize_slice_img(fix_1, jdet_dir + 'fix_1.png', idx1)
visualize_slice_img(fix_seg_1, jdet_dir + 'fix_seg_1.png', idx1)
visualize_slice_img(mov_1, jdet_dir + 'mov_1.png', idx1)
visualize_slice_img(mov_seg_1, jdet_dir + 'mov_seg_1.png', idx1)

visualize_slice_img(w_img_0, jdet_dir + 'vxm_w_img_0.png', idx0)
visualize_slice_img(w_img_1, jdet_dir + 'vxm_w_img1.png', idx1)
visualize_slice_img(w_seg_0, jdet_dir + 'vxm_w_seg_0.png', idx0)
visualize_slice_img(w_seg_1, jdet_dir + 'vxm_w_seg_1.png', idx1)

print('JDet paths', jdet_paths)
print('Img0 paths', img0_paths)
print('Seg0 paths', seg0_paths)
print('Img1 paths', img1_paths)
print('Seg1 paths', seg1_paths)




# VTN
directory = '/data/groups/beets-tan/l.estacio/Med_Align_Net/Jul10-004547_lutrain_VTNx3___/'

image_paths   = get_file_paths(directory)
jdet_paths    = image_paths['jdet.nii.gz']
jdet_0_path = jdet_paths[0] # 31_34 worst
jdet_1_path = jdet_paths[1] # 35_40 better
print('len: ', len(jdet_paths))

color_range = (0, 2)
img, _ = load_nifti_image(jdet_0_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'vtn_0.png', idx0, color_range=color_range, cmap='seismic')

color_range        = (0, 2)
img, _ = load_nifti_image(jdet_1_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'vtn_1.png', idx1, color_range=color_range, cmap='seismic')

# Images
w_img_paths  = image_paths['w_img.nii.gz']
w_img_0_path = w_img_paths[0]
w_img_1_path = w_img_paths[1]
w_seg_paths  = image_paths['w_seg.nii.gz']
w_seg_0_path = w_seg_paths[0]
w_seg_1_path = w_seg_paths[1]


w_img_0, _ = load_nifti_image(w_img_0_path)
w_img_1, _ = load_nifti_image(w_img_1_path)
w_seg_0, _ = load_nifti_image(w_seg_0_path)
w_seg_1, _ = load_nifti_image(w_seg_0_path)

visualize_slice_img(w_img_0, jdet_dir + 'vtn_w_img_0.png', idx0)
visualize_slice_img(w_img_1, jdet_dir + 'vtn_w_img1.png', idx1)
visualize_slice_img(w_seg_0, jdet_dir + 'vtn_w_seg_0.png', idx0)
visualize_slice_img(w_seg_1, jdet_dir + 'vtn_w_seg_1.png', idx1)






# TSM

directory = '/data/groups/beets-tan/l.estacio/Med_Align_Net/Jul10-164631_lutrain_TSMx1___/'

image_paths   = get_file_paths(directory)
jdet_paths    = image_paths['jdet.nii.gz']
jdet_0_path = jdet_paths[0] # 31_34 worst
jdet_1_path = jdet_paths[1] # 35_40 better
print('len: ', len(jdet_paths))

color_range = (0, 2)
img, _ = load_nifti_image(jdet_0_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'tsm_0.png', idx0, color_range=color_range, cmap='seismic')

color_range        = (0, 2)
img, _ = load_nifti_image(jdet_1_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'tsm_1.png', idx1, color_range=color_range, cmap='seismic')


# Imgs
w_img_paths  = image_paths['w_img.nii.gz']
w_img_0_path = w_img_paths[0]
w_img_1_path = w_img_paths[1]
w_seg_paths  = image_paths['w_seg.nii.gz']
w_seg_0_path = w_seg_paths[0]
w_seg_1_path = w_seg_paths[1]


w_img_0, _ = load_nifti_image(w_img_0_path)
w_img_1, _ = load_nifti_image(w_img_1_path)
w_seg_0, _ = load_nifti_image(w_seg_0_path)
w_seg_1, _ = load_nifti_image(w_seg_0_path)

visualize_slice_img(w_img_0, jdet_dir + 'tsm_w_img_0.png', idx0)
visualize_slice_img(w_img_1, jdet_dir + 'tsm_w_img1.png', idx1)
visualize_slice_img(w_seg_0, jdet_dir + 'tsm_w_seg_0.png', idx0)
visualize_slice_img(w_seg_1, jdet_dir + 'tsm_w_seg_1.png', idx1)




# CLMorph

directory = '/data/groups/beets-tan/l.estacio/Med_Align_Net/Aug08-194443_lutrain_ALNx1___/'

image_paths   = get_file_paths(directory)
jdet_paths    = image_paths['jdet.nii.gz']
jdet_0_path = jdet_paths[0] # 31_34 worst
jdet_1_path = jdet_paths[1] # 35_40 better
print('len: ', len(jdet_paths))

color_range = (0, 2)
img, _ = load_nifti_image(jdet_0_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'CLM_0.png', idx0, color_range=color_range, cmap='seismic')

color_range        = (0, 2)
img, _ = load_nifti_image(jdet_1_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'CLM_1.png', idx1, color_range=color_range, cmap='seismic')


# Imgs
w_img_paths  = image_paths['w_img.nii.gz']
w_img_0_path = w_img_paths[0]
w_img_1_path = w_img_paths[1]
w_seg_paths  = image_paths['w_seg.nii.gz']
w_seg_0_path = w_seg_paths[0]
w_seg_1_path = w_seg_paths[1]


w_img_0, _ = load_nifti_image(w_img_0_path)
w_img_1, _ = load_nifti_image(w_img_1_path)
w_seg_0, _ = load_nifti_image(w_seg_0_path)
w_seg_1, _ = load_nifti_image(w_seg_0_path)

visualize_slice_img(w_img_0, jdet_dir + 'CLM_w_img_0.png', idx0)
visualize_slice_img(w_img_1, jdet_dir + 'CLM_w_img1.png', idx1)
visualize_slice_img(w_seg_0, jdet_dir + 'CLM_w_seg_0.png', idx0)
visualize_slice_img(w_seg_1, jdet_dir + 'CLM_w_seg_1.png', idx1)




# Elastix

directory = '/data/groups/beets-tan/l.estacio/Med_Align_Net/elastix/'

image_paths   = get_file_paths(directory)
jdet_paths    = image_paths['jdet.nii.gz']
jdet_0_path = jdet_paths[0] # 31_34 worst
jdet_1_path = jdet_paths[1] # 35_40 better
print('len: ', len(jdet_paths))

color_range = (0, 2)
img, _ = load_nifti_image(jdet_0_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'elastix_0.png', idx0, color_range=color_range, cmap='seismic')

color_range        = (0, 2)
img, _ = load_nifti_image(jdet_1_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'elastix_1.png', idx1, color_range=color_range, cmap='seismic')


# Imgs
w_img_paths  = image_paths['w_img.nii.gz']
w_img_0_path = w_img_paths[0]
w_img_1_path = w_img_paths[1]
w_seg_paths  = image_paths['w_seg.nii.gz']
w_seg_0_path = w_seg_paths[0]
w_seg_1_path = w_seg_paths[1]


w_img_0, _ = load_nifti_image(w_img_0_path)
w_img_1, _ = load_nifti_image(w_img_1_path)
w_seg_0, _ = load_nifti_image(w_seg_0_path)
w_seg_1, _ = load_nifti_image(w_seg_0_path)

visualize_slice_img(w_img_0, jdet_dir + 'elastix_w_img_0.png', idx0)
visualize_slice_img(w_img_1, jdet_dir + 'elastix_w_img1.png', idx1)
visualize_slice_img(w_seg_0, jdet_dir + 'elastix_w_seg_0.png', idx0)
visualize_slice_img(w_seg_1, jdet_dir + 'elastix_w_seg_1.png', idx1)


# ANTs

directory = '/data/groups/beets-tan/l.estacio/Med_Align_Net/ants/'

image_paths   = get_file_paths(directory)
jdet_paths    = image_paths['jdet.nii.gz']
jdet_0_path = jdet_paths[0] # 31_34 worst
jdet_1_path = jdet_paths[1] # 35_40 better
print('len: ', len(jdet_paths))

color_range = (0, 2)
img, _ = load_nifti_image(jdet_0_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'ants_0.png', idx0, color_range=color_range, cmap='seismic')

color_range        = (0, 2)
img, _ = load_nifti_image(jdet_1_path)
print(img.shape, ' min: ', img.min(), ' max: ', img.max())
visualize_jacobian(img, jdet_dir + 'ants_1.png', idx1, color_range=color_range, cmap='seismic')


# Imgs
w_img_paths  = image_paths['w_img.nii.gz']
w_img_0_path = w_img_paths[0]
w_img_1_path = w_img_paths[1]
w_seg_paths  = image_paths['w_seg.nii.gz']
w_seg_0_path = w_seg_paths[0]
w_seg_1_path = w_seg_paths[1]


w_img_0, _ = load_nifti_image(w_img_0_path)
w_img_1, _ = load_nifti_image(w_img_1_path)
w_seg_0, _ = load_nifti_image(w_seg_0_path)
w_seg_1, _ = load_nifti_image(w_seg_0_path)

visualize_slice_img(w_img_0, jdet_dir + 'ants_w_img_0.png', idx0)
visualize_slice_img(w_img_1, jdet_dir + 'ants_w_img1.png', idx1)
visualize_slice_img(w_seg_0, jdet_dir + 'ants_w_seg_0.png', idx0)
visualize_slice_img(w_seg_1, jdet_dir + 'ants_w_seg_1.png', idx1)