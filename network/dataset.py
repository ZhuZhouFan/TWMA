import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
# from torchvision.io.image import read_image
from PIL import Image
# from torchvision.transforms.functional import normalize, resize
import numpy as np
from torchvision.transforms import transforms

my_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CV_Dataset(Dataset):
    def __init__(self,
                 data_dir:str,
                 start_time:str,
                 end_time:str,
                 lag_order:int,
                 horizon:int,
                 data_info:str = None):
        super(CV_Dataset, self).__init__()
        
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.lag_order = lag_order
        self.horizon = horizon
        self.label_list = []
        
        if data_info is None:
            self.data_info = self.get_data_info()
        else:
            self.data_info = data_info

    def __getitem__(self, index):
        data_path = self.data_info[index]
        img = Image.open(f'{data_path}/I{self.lag_order}.jpeg')
        feature = my_transform(img)
        label = np.load(f'{data_path}/R{self.horizon}.npy')
        if label > 0:
            label = torch.LongTensor([1])
        else:
            label = torch.LongTensor([0])
        
        return feature, label

    def __len__(self):
        return len(self.data_info)

    def get_data_info(self):
        data_info = list()
        date_list = os.listdir(self.data_dir)
        date_list.sort()
        valid_date_list = list(filter(lambda x: (x >= self.start_time) & (x <= self.end_time),
                                      date_list))
        for date in tqdm(valid_date_list, desc = 'preparing for data pipeline'):
            date_path = os.path.join(self.data_dir, date)
            for root, dirs, files in os.walk(date_path):
                for dir_ in dirs:
                    feature_path = os.path.join(date_path, dir_, f'I{self.lag_order}.jpeg')
                    label_path = os.path.join(date_path, dir_, f'R{self.horizon}.npy')
                    if os.path.exists(feature_path) & os.path.exists(label_path):
                        data_info.append(os.path.join(date_path, dir_))
        return data_info 
    
class Inference_Dataset(Dataset):
    def __init__(self,
                 lag_order:int,
                 horizon:int,
                 data_info:str):
        super(Inference_Dataset, self).__init__()
        
        self.lag_order = lag_order
        self.horizon = horizon
        self.data_info = data_info            

    def __getitem__(self, index):
        data_path = self.data_info[index]
        date, stock = data_path.split('/')[-2:]
        try:
            img = Image.open(f'{data_path}/I{self.lag_order}.jpeg')
            feature = my_transform(img)
            label = np.load(f'{data_path}/R{self.horizon}.npy').item()
            
            if (feature.shape[0] != 3):
                feature = feature.expand(3, -1, -1)
            
            if not np.isscalar(label):
                label = 0.0
        except Exception as e:
            feature = torch.zeros(3, 224, 224)
            label = 0.0
        return date, stock, feature, label

    def __len__(self):
        return len(self.data_info)