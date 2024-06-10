import json
from torch.utils.data import Dataset

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
    
        #self.data = self.add_root_dir_to_paths()
    
    
    def get_pairs_with_gt(self, data):
        #fix_info = data[:20]
        #mov_info = data[20:]
        #return {'fix_info': fix_info, 'mov_info': mov_info}
        # Calculate the midpoint index
        midpoint_index = len(data) // 2

        # Initialize lists to store pairs
        pairs = []

        # Create pairs from the first half and the second half
        for i in range(midpoint_index):
            pair = {
                'fix_00' + str(i + 1): data[i],
                'mov_00' + str(i + 1): data[i + midpoint_index]
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
            
    
    def __len__(self):
        
        print(self.data)
        return len(self.data)
    
    def __getitem__(self, index: int):
        pass
        
 
class Data(RawData, Dataset):
    def __init__(self, args, **kwargs):
        RawData.__init__(self, args, **kwargs)
        

import sys 
sys.path.append('..')
from tools.utils import show_img
import matplotlib.pyplot as plt
#if __name__  == 'main':
data_file = '/home/cerquinl/projects/raw_data/LungCT/LungCT_dataset.json'
root_dir  = '/processing/l.estacio/L2R-Dataset-2022/LungCT'


data      = Data(data_file, root_dir=root_dir, mode='train')
print(len(data))
print(data)
    