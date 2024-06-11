#%%
import json
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from processing.cts_processors import ScanProcessor
from processing.cts_operations import ReadVolume
from processing.cts_operations import PadAndCropTo
from processing.cts_operations import ToLungWindowLevelNormalization
# %%
class Split:
    TRAIN = 1
    VALID = 2


class RawData():
    def __init__(self, json_file, root_dir, mode='train', transform=None):
        with open(json_file, 'r') as file:
            data_info = json.load(file)
        self.root_dir  = root_dir
        self.mode      = mode
        self.transform = transform
        self.inp_dtype = torch.float32
        self.loader    = self.__init_operations()
        
        
        mode_mapping = {
            'train': 'training',
            'val'  : 'registration_val',
            'test' : 'test'
        }
        
        if self.mode in mode_mapping:
            data = data_info[mode_mapping[self.mode]]
        else:
            raise ValueError('mode can only be train, val, or test')
        
        if self.mode != 'val':
            self.data = self.get_pairs_with_gt(data)
        else: self.data = data
    
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
            
    def __init_operations(self):
        return ScanProcessor(
            ReadVolume(),
            ToLungWindowLevelNormalization()
        )
    
    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, idx: int):
        #print(self.data)
        ret = {}
        #print('Hello: ', self.data[idx]['fix']['image'])
        '''ret['voxel1'] = torch.from_numpy(self.loader(self.data[idx]['fix']['image'])).type(self.inp_dtype)
        ret['voxel2'] = torch.from_numpy(self.loader(self.data[idx]['mov']['image'])).type(self.inp_dtype)
        ret['segmentation1'] = torch.from_numpy(self.loader(self.data[idx]['fix']['mask'])).type(self.inp_dtype)
        ret['segmentation2'] = torch.from_numpy(self.loader(self.data[idx]['mov']['mask'])).type(self.inp_dtype)'''
        ret['voxel1'] = self.loader(self.data[idx]['fix']['image'])
        ret['voxel2'] = self.loader(self.data[idx]['mov']['image'])
        ret['segmentation1'] = self.loader(self.data[idx]['fix']['mask'])
        ret['segmentation2'] = self.loader(self.data[idx]['mov']['mask'])
        #print(ret) 
        return ret
        
        
        
 
class Data(RawData, Dataset):
    def __init__(self, args, **kwargs):
        RawData.__init__(self, args, **kwargs)
        



import sys 
sys.path.append('..')
from tools.utils         import show_img
from tools.visualization import plot_sample_data

import matplotlib.pyplot as plt
#if __name__  == 'main':
data_file = '/home/cerquinl/projects/raw_data/LungCT/LungCT_dataset.json'
root_dir  = '/home/cerquinl/projects/raw_data/LungCT/'


data      = Data(data_file, root_dir=root_dir, mode='train')
print(len(data))
plot_sample_data(data[0], slide=128, save_path='./128.png')
    