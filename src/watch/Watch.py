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
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=3)

    def forward(self, x):
        '''

        @param x: Input tensor
        @return: Stacked outputs from LSTM and the final hidden state
        '''

        num_channels = x.size(2)
        batch_size = x.size(0)
        feats = []

        for i in range(num_channels - 5 + 1):
            feat = x[:, :, i:i+5, :, :]
            feat = self.sync_net.forward(feat)
            feat = feat.view(batch_size, 1, -1)
            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        output_state, (hidden, carry) = self.lstm(feats)

        return output_state, hidden

    def get_model(self):
        ## TODO
        ## Build LSTM Model here
        ## Add CNN to the same model
        return self.sync_net

    def get_parameters(self):
        return self.sync_net.parameters()
