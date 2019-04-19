import torch
from torch.nn.utils.rnn import pad_sequence

from LRWDataset import LRWDataset
from torch.utils.data import DataLoader

def collate_data_streams(batch):
    mp4_data = []
    mp3_data = []
    txt_data = []
    for i in range(len(batch)):
        mp4_data.append(batch[i][0])
        mp3_data.append(batch[i][1])
        txt_data.append(batch[i][2])
    mp4_pad = pad_sequence(mp4_data, batch_first=True)
    # mp3_pad = pad_sequence(mp3_data, batch_first=True)
    # txt_pad = pad_sequence(txt_data, batch_first=True)
    return mp4_pad, mp3_data, txt_data # _pad, txt_pad

if __name__ == '__main__':
    root_dir = '/Users/sri/P/audio-assisted-lip-reading/data/'
    dev_dir = 'dev/'
    dataset = LRWDataset(root_dir, dev_dir + 'dev.csv', is_dev=True)
    batch_size = 4

    dataloader = DataLoader(dataset,
                            collate_fn=collate_data_streams,
                            batch_size=batch_size,
                            drop_last=True)

    for mp4, mp3, txt in dataloader:
        print(mp4.shape)
        print(mp3)
        print(txt)
        assert False