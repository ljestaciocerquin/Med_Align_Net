import os
import glob
import numpy     as np
import SimpleITK as sitk
from   tqdm      import tqdm
'''
Abdominal images are flipped horizontally, and when we run TotalSegmentator we did not get
proper segmentations of the liver. Therefore, this class, was done to get the data on the right view.
'''

class Horizontal_Flip(object):
    def __init__(self):
        super(Horizontal_Flip, self).__init__()
        
        
    def read_data(self, input_folder):
        nii_gz_paths = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.nii.gz'):
                    # Append the full path to the list
                    nii_gz_paths.append(os.path.join(root, file))
        return nii_gz_paths


    def make_folder_to_save(self, output_folder):
        if not os.path.exists(output_folder):
            print('Creating folder...')
            os.makedirs(output_folder)
        else:
            print('This folder already exists :)!')
            
            
    def get_segmentations(self, data, output_folder):
        
        with tqdm(total=len(data)) as pbar:
            #import pdb;
            #pdb.set_trace()
            for scan in data:
                
                image       = sitk.ReadImage(scan)
                image_array = sitk.GetArrayFromImage(image)

                # Flip the image array horizontally
                flipped_array = np.flip(image_array, axis=-1)
                flipped_image = sitk.GetImageFromArray(flipped_array)

                # Copy the metadata (origin, spacing, direction) from the original image
                flipped_image.SetOrigin(image.GetOrigin())
                flipped_image.SetSpacing(image.GetSpacing())
                flipped_image.SetDirection(image.GetDirection())

                # Save the flipped image to a new file
                new_path = output_folder + scan.split('/')[-2] + '.nii.gz'
                print(new_path)
                sitk.WriteImage(flipped_image, new_path)
                
                pbar.update(1)



if __name__ == "__main__":
    horflip = Horizontal_Flip()
    input_folder  = '/data/groups/beets-tan/l.estacio/abdomen_data/AbdomenCTCT/labels_ts/'
    output_folder = '/data/groups/beets-tan/l.estacio/abdomen_data/AbdomenCTCT/labelsTS/'
    
    horflip.make_folder_to_save(output_folder)
    data = horflip.read_data(input_folder)
    horflip.get_segmentations(data, output_folder)