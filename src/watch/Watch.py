import watch.SyncNetModel as SNet

class WatchNet:
    def __init__(self, root_dir, model_path):
        '''

        @param root_dir: Root dir of the entire project
        @param model_path: Path to the model, relative to the root dir
        '''
        self.root_dir = root_dir
        self.model_path = model_path
        self.sync_net = SNet.S()
        net = SNet.load(self.root_dir + self.model_path)
        self.sync_net.load_state_dict(net)

    def forward(self, x):
        '''

        @param x: Input tensor
        @return: Features of running input through CNN
        '''
        x = self.sync_net.forward(x)
        return x

    def get_model(self):
        return self.sync_net

    def get_parameters(self):
        return self.sync_net.parameters()
