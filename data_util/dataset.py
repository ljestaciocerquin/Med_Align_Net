#%%
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from processing.cts_processors import ScanProcessor
from processing.cts_operations import ReadVolume
from processing.cts_operations import PadAndCropTo
from processing.cts_operations import ToLungWindowLevelNormalization

class RawData():
    def __init__(self, json_file, root_dir, mode='train', transform=None):
        with open(json_file, 'r') as file:
            data_info = json.load(file)
        self.root_dir  = root_dir
        self.mode      = mode
        self.transform = transform
        self.inp_dtype = torch.float32
        self.loader    = self.__init_loader()
        self.loader_op = self.__init_operations()
        
        # We didn't consider registration_val since they are the three first elements of the training dataset
        # This 3 elements have landmarks information
        mode_mapping = {
            'train': 'training',
            'val':   'training',
            'test' : 'test'
        }
        
        if self.mode in mode_mapping:
            data = data_info[mode_mapping[self.mode]]
        else:
            raise ValueError('mode can only be train, or test')
        
        self.data = self.get_pairs_with_gt(data)
        if self.mode=='val': 
            self.data = self.data[:3]
            
        self.add_root_dir_to_paths()
    
    
    
    def get_pairs_with_gt(self, data):
        # Calculate the midpoint index
        midpoint_index = len(data) // 2
        pairs = []

        # Create pairs from the first half and the second half
        for i in range(midpoint_index):
            pair = {
                'fix': data[i],
                'mov': data[i + midpoint_index]
            }
            pairs.append(pair)
        return pairs
    
    
    def add_root_dir_to_paths(self):
        for pair in self.data:
            for key, value in pair.items():
                # Iterate through each key-value pair in the nested dictionary
                for item_key, item_value in value.items():
                    # Update the path with the new project path
                    pair[key][item_key] = self.root_dir + item_value.lstrip('./')
    
                    
    
    def read_keypoints(self, file):
        kps = pd.read_csv(file, header=None).values.astype(int)
        return kps
        
    def __init_loader(self):
        return ScanProcessor(
            ReadVolume(),
        )
        
              
    def __init_operations(self):
        return ScanProcessor(
            ReadVolume(),
            ToLungWindowLevelNormalization()
        )
    
    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, idx: int):
        
        ret = {}
        if self.mode == 'train':
            ret['img1_path']     = self.data[idx]['fix']['image']
            ret['img2_path']     = self.data[idx]['mov']['image']
            voxel1        = torch.from_numpy(self.loader_op(self.data[idx]['fix']['image'])).type(self.inp_dtype)
            voxel2        = torch.from_numpy(self.loader_op(self.data[idx]['mov']['image'])).type(self.inp_dtype)
            segmentation1 = torch.from_numpy(self.loader_op(self.data[idx]['fix']['mask'])).type(self.inp_dtype)
            segmentation2 = torch.from_numpy(self.loader_op(self.data[idx]['mov']['mask'])).type(self.inp_dtype)
            kps1          = torch.from_numpy(self.read_keypoints(self.data[idx]['fix']['keypoints']))
            kps2          = torch.from_numpy(self.read_keypoints(self.data[idx]['mov']['keypoints']))
            ret['voxel1']        = voxel1[None, :]
            ret['voxel2']        = voxel2[None, :]
            ret['segmentation1'] = segmentation1[None, :]
            ret['segmentation2'] = segmentation2[None, :]
            ret['kps1']          = kps1
            ret['kps2']          = kps2
            
        
        else:
            ret['img1_path']     = self.data[idx]['fix']['image']
            ret['img2_path']     = self.data[idx]['mov']['image']
            #ret['img1']          = self.loader(self.data[idx]['fix']['image'])
            #ret['img2']          = self.loader(self.data[idx]['mov']['image'])
            ret['voxel1']        = torch.from_numpy(self.loader_op(self.data[idx]['fix']['image'])).type(self.inp_dtype)
            ret['voxel2']        = torch.from_numpy(self.loader_op(self.data[idx]['mov']['image'])).type(self.inp_dtype)
            ret['segmentation1'] = torch.from_numpy(self.loader_op(self.data[idx]['fix']['mask'])).type(self.inp_dtype)
            ret['segmentation2'] = torch.from_numpy(self.loader_op(self.data[idx]['mov']['mask'])).type(self.inp_dtype)
            ret['kps1']          = torch.from_numpy(self.read_keypoints(self.data[idx]['fix']['keypoints']))
            ret['kps2']          = torch.from_numpy(self.read_keypoints(self.data[idx]['mov']['keypoints']))
            ret['lmk1']          = torch.from_numpy(self.read_keypoints(self.data[idx]['fix']['landmarks']))
            ret['lmk2']          = torch.from_numpy(self.read_keypoints(self.data[idx]['mov']['landmarks']))
        #print(ret) 
        return ret
        
        
        
 
class Data(RawData, Dataset):
    def __init__(self, args, **kwargs):
        RawData.__init__(self, args, **kwargs)
        


#%%
import sys 
sys.path.append('..')
from tools.utils         import show_img
from tools.visualization import plot_sample_data

import matplotlib.pyplot as plt
if __name__  == '__main__':
    data_file = '/data/groups/beets-tan/l.estacio/lung_data/LungCT/LungCT_dataset.json'
    root_dir  = '/data/groups/beets-tan/l.estacio/lung_data/LungCT/'


    data      = Data(data_file, root_dir=root_dir, mode='val')
    print(len(data))
    plot_sample_data(data[0], slide=128, save_path='./128_.png')
    #print(data[0])

# %%
