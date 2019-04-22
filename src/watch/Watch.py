import torch
import torch.nn as nn
import watch.SyncNetModel as SNet

class WatchNet:
    def __init__(self, root_dir, model_path, device):
        '''

        @param root_dir: Root dir of the entire project
        @param model_path: Path to the model, relative to the root dir
        '''
        self.root_dir = root_dir
        self.model_path = model_path
        self.sync_net = SNet.S()
        net = SNet.load(self.root_dir + self.model_path, device)
        self.sync_net.load_state_dict(net)
        self.sync_net = self.sync_net.to(device)

    def forward(self, x):
        '''

        @param x: Input tensor
        @return: Features of running input through CNN
        '''
        # b_size, frames, h, w, channels = x.size()
        # x = x.view(b_size, channels, frames, h, w)
        # print(x.size())
        # test = x[:, : ,4:9]
        #

        num_channels = x.size(2)
        for i in range(num_channels - 5 + 1):
            feat = x[:, :, i:i+5, :, :]
            feat = self.sync_net.forward(feat)
            
        x = self.sync_net.forward(x)
        return x

    def get_model(self):
        ## TODO
        ## Build LSTM Model here
        ## Add CNN to the same model
        return self.sync_net

    def get_parameters(self):
        return self.sync_net.parameters()
