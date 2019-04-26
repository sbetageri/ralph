import torch
import torch.optim

from watch import Watch
from listen import Listen
from spell import Spell
from attention import Attention

from torch.nn.utils.rnn import pad_sequence

from LRWDataset import LRWDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
import Levenshtein as Lev

SPELL_LAYERS = 3
SPELL_HIDDEN = 256
SPELL_OUTPUT = 40

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
    txt_pad = pad_sequence(txt_data, batch_first=True, padding_value=38.0)
    txt_pad = txt_pad.long()
    return mp4_pad, mp3_pad, txt_pad# _pad, txt_pad

def reshape_mp4_tensors(mp4):
    b_size, frames, h, w, channels = mp4.size()
    mp4 = mp4.view(b_size, channels, frames, h, w)
    return mp4

def pad_text(txt, device):
    diff = 40 - txt.size(1)
    z = torch.zeros((batch_size, diff)).long()
    z = z.to(device)
    return torch.cat((txt, z), dim=1)


if __name__ == '__main__':
    root_dir = '../data/'
    dev_dir = 'dev/'
    model_path = 'syncnet_v2.model'
    dataset = LRWDataset(root_dir, dev_dir + 'dev.csv', is_dev=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    watch_net = Watch.WatchNet(root_dir, model_path, device)
    listen_net = Listen.ListenNet(device)
    spell_net = Spell.SpellNet(SPELL_LAYERS, SPELL_HIDDEN, SPELL_OUTPUT, device)

    # listen_model = listen.get_model()
    # spell_model = spell.get_model()

    dataloader = DataLoader(dataset,
                            collate_fn=collate_data_streams,
                            batch_size=batch_size,
                            drop_last=True)

    watch_param = watch_net.parameters()
    listen_param = listen_net.parameters()
    spell_param = spell_net.parameters()

    tot_param = list(watch_param) + list(listen_param) + list(spell_param)
    optimizer = torch.optim.Adam(tot_param, lr=0.1, amsgrad=True)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()

    tot_loss = 0
    for e in range(10):
        running_loss = 0
        for mp4, mp3, txt in tqdm(dataloader):

            ## Move from CPU to GPU, if needed
            optimizer.zero_grad()
            mp4 = mp4.to(device)
            mp3 = mp3.to(device)
            txt = txt.to(device)

            video_out, video_hidden = watch_net.forward(mp4)
            audio_out, audio_hidden = listen_net.forward(mp3)

            video_hidden = video_hidden.view(1, *video_hidden.size())
            audio_hidden = audio_hidden.view(1, *audio_hidden.size())

            av_state = torch.cat((video_hidden, audio_hidden))

            # Reshaped this way specifically.
            ## NEED TO KEEP BATCH_SIZE = 2
            ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            av_state = av_state.view(3, 2, -1)

            cell_state = torch.zeros_like(av_state).to(device)
            out = spell_net.forward(txt, av_state, cell_state, video_out, audio_out)

            # spell_out = spell_model(txt, video_out, audio_out, l1_out, l2_out, l3_out)
            # loss = criterion(spell_out, txt)
            t = pad_text(txt.item(), device).float()
            loss = criterion(out[0], t)
            optimizer.step()
            running_loss += loss.item()
            # txt.expand(batch_size, 40)
        tot_loss += running_loss / (len(dataloader)) * 4
    tot_loss /= 10
    print(tot_loss)



