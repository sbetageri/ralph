import torch
import imageio

from PIL import Image
from torch.utils.data import Dataset


class LRWDataset(Dataset):
    COL_MP4 = 'mp4'
    COL_MP3 = 'mp3'
    COL_TXT = 'txt'
    
    def __init__(self, root_dir, clean_files_path, is_train=True):
        self.root_dir = root_dir
        self.df = get_files(root_dir, clean_files_path, is_train)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        mp4, mp3, txt = self._get_records(idx)
        
        reversed_mp3 = self._get_reversed_mp3_as_tensor(self.root_dir + mp3)
        reversed_txt = self._get_reversed_txt_as_tensor(self.root_dir + txt)
        reversed_mp4 = self._get_reversed_frames_as_tensors(self.root_dir + mp4)
        
        return reversed_mp4, reversed_mp3, reversed_txt
    
    def _get_files(self, root_dir, file_path, is_train = True):
        df = pd.read_csv(root_dir + file_path)
        if is_train:
            return df[df['is_train'] == 1]
        else:
            return df[df['is_train'] == 0]
    
    def _get_records(self, idx):
        record = df.iloc[idx]
        mp4 = record[LRWDataset.COL_MP4]
        mp3 = record[LRWDataset.COL_MP3]
        txt = record[LRWDataset.COL_TXT]
        
        return mp4, mp3, txt
    
    def _get_reversed_mp3_as_tensor(self, mp3_path):
        return mp3_path
    
    def _get_reversed_txt_as_tensor(self, txt_path):
        return txt_path
    
    def _get_reversed_frames_as_tensors(self, mp4_file):
        reader = imageio.get_reader(mp4_file)
        reader = imageio.get_reader(mp4_file)
        imgs = np.array(reader.get_data(0))
        imgs = imgs.reshape(1, *imgs.shape)
        count = reader.count_frames()
        for i in range(1, count):
            frame = np.array(reader.get_data(i))
            frame = frame.reshape(1, *frame.shape)
            imgs = np.vstack((imgs, frame))
        frames = torch.from_numpy(imgs)
        rev_frames = torch.flip(frames, [0])
        return rev_frames