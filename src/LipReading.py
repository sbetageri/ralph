import torch
import listen

from watch import Watch
from listen import Listen

from torch.nn.utils.rnn import pad_sequence

from LRWDataset import LRWDataset
from torch.utils.data import DataLoader

def collate_data_streams(batch):
    mp4_data = []
    mp3_data = []
    txt_data = []
    for i in range(len(batch)):
        mp4_data.append(batch[i][0])
        mp3_data.append(batch[i][1].transpose(0, 1))
        txt_data.append(batch[i][2])
    mp4_pad = pad_sequence(mp4_data, batch_first=True)
    mp3_pad = pad_sequence(mp3_data, batch_first=True)

    mp4_pad = reshape_mp4_tensors(mp4_pad)
    # txt_pad = pad_sequence(txt_data, batch_first=True)
    return mp4_pad, mp3_pad, txt_data # _pad, txt_pad

def reshape_mp4_tensors(mp4):
    b_size, frames, h, w, channels = mp4.size()
    mp4 = mp4.view(b_size, channels, frames, h, w)
    return mp4


if __name__ == '__main__':
    root_dir = '../data/'
    dev_dir = 'dev/'
    model_path = 'syncnet_v2.model'
    dataset = LRWDataset(root_dir, dev_dir + 'dev.csv', is_dev=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    watch_net = Watch.WatchNet(root_dir, model_path, device)
    listen_net = Listen.ListenNet(device)

    # listen_model = listen.get_model()
    # spell_model = spell.get_model()

    dataloader = DataLoader(dataset,
                            collate_fn=collate_data_streams,
                            batch_size=batch_size,
                            drop_last=True)

    watch_param = watch_net.get_parameters()
    listen_param = listen_net.get_parameters()
    # spell_param = spell.get_parameters()

    # tot_param = list(watch_param) + list(listen_param) + list(spell_param)
    # optimizer = torch.optim.sgd(tot_param, lr=0.01)
    # criterion = torch.nn.CrossEntropyLoss()

    for mp4, mp3, txt in dataloader:

        ## Move from CPU to GPU, if needed
        mp4 = mp4.to(device)
        mp3 = mp3.to(device)

        video_out, video_states = watch_net.forward(mp4)
        audio_out, audio_states = listen_net.forward(mp3)
        # print(video_out.size())
        # print(video_states.size())
        # audio_out, la1_out, la2_out, la3_out = listen_model(mp3)

        # spell_out = spell_model(txt, video_out, audio_out, l1_out, l2_out, l3_out)
        # loss = criterion(spell_out, txt)

        assert False
