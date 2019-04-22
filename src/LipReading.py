import torch
import listen

from watch import Watch
from listen import Listen
from spell import Spell
from attention import Attention

from torch.nn.utils.rnn import pad_sequence

from LRWDataset import LRWDataset
from torch.utils.data import DataLoader
import Levenshtein as Lev

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

def trainIters(args):
	'''
	FOR REFERENCE
	'''
	watch = Watch(3, 512, 512)
	spell = Spell(3, 512, LRWDataset.total_num)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	watch = watch.to(device)
	spell = spell.to(device)

	watch_optimizer = optim.Adam(watch.parameters(),
	                             lr=args['LEARNING_RATE'])
	spell_optimizer = optim.Adam(spell.parameters(),
	                             lr=args['LEARNING_RATE'])
	watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=args['LEARNING_RATE_DECAY_EPOCH'],
	                                            gamma=args['LEARNING_RATE_DECAY_RATIO'])
	spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=args['LEARNING_RATE_DECAY_EPOCH'],
	                                            gamma=args['LEARNING_RATE_DECAY_RATIO'])
	criterion = nn.CrossEntropyLoss(ignore_index=get_charSet().get_index_of('<pad>'))

	train_loader, eval_loader = get_dataloaders(args['PATH'], args['BS'], args['VMAX'], args['TMAX'], args['WORKER'],
	                                            args['VALIDATION_RATIO'])

	total_batch = len(train_loader)
	total_eval_batch = len(eval_loader)

	for epoch in range(args['ITER']):
		avg_loss = 0.0
		avg_eval_loss = 0.0
		avg_cer = 0.0
		avg_eval_cer = 0.0
		watch_scheduler.step()
		spell_scheduler.step()

		watch = watch.train()
		spell = spell.train()

		for i, (data, length, labels) in enumerate(train_loader):
            results = []
            data = data.to(device)
            labels = labels.to(device)
            length = length.to(device)

            watch_optimizer.zero_grad()
            spell_optimizer.zero_grad()

            target_length = labels.size(1)
            loss = 0
            cer = 0

            watch_outputs, watch_state = watch(data, length)

            spell_input = torch.tensor([[LRWDataset.index_of_str['<sos>']]]).repeat(watch_outputs.size(0), 1).to(device)
            spell_hidden = watch_state
            cell_state = torch.zeros_like(spell_hidden).to(device)
            context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2)).to(device)

            for di in range(target_length):
                spell_output, spell_hidden, cell_state, context = spell(spell_input, spell_hidden, cell_state,
                                                                        watch_outputs, context)
                _, topi = spell_output.topk(1, dim=2)
                spell_input = labels[:, di].long().unsqueeze(1)

                loss += criterion(spell_output.squeeze(1), labels[:, di].long())
                results.append(topi.cpu().squeeze(1))

            loss = loss.to(device)
            loss.backward()

            watch_optimizer.step()
            spell_optimizer.step()

            results = torch.cat(results, dim=1)
            for batch in range(results.size(0)):
                output = ''
                label = ''
                for index in range(target_length):
                    output += LRWDataset.char_of_index[int(results[batch, index])]
                    label += LRWDataset.char_of_index[int(labels[batch, index])]
                label = label.replace('<pad>', '').replace('<eos>', '@')
                output = output.replace('<eos>', '@')[:output.find('@')].replace('<pad>', '$').replace('<sos>', '&')
                cer += Lev.distance(output, label)

            loss = loss.item() / target_length
            avg_loss += loss
            avg_cer += cer
            print('Batch : ', i + 1, '/', total_batch, ', ERROR in this minibatch: ', loss)
            print('Character error rate : ', cer)

        watch = watch.eval()
        spell = spell.eval()

        for k, (data, length, labels) in enumerate(eval_loader):
            loss, cer = train(data, length, labels, watch, spell, watch_optimizer, spell_optimizer, criterion, False)
            avg_eval_loss += loss
            avg_eval_cer += cer

        print('epoch:', epoch, ' train_loss:', float(avg_loss / total_batch))
        print('epoch:', epoch, ' Average CER:', float(avg_cer / total_batch))
        print('epoch:', epoch, ' Validation_loss:', float(avg_eval_loss / total_eval_batch))
        print('epoch:', epoch, ' Average CER:', float(avg_eval_cer / total_eval_batch))
        if epoch % args['SAVE_EVERY'] == 0 and epoch != 0:
            torch.save(watch, 'watch{}.pt'.format(epoch))
            torch.save(spell, 'spell{}.pt'.format(epoch))