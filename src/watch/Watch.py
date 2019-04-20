import watch.SyncNetModel as SNet

class Watch:
    def __init__(self, root_dir, model_path):
        '''

        @param root_dir: Root dir of the entire proj
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
        x = self.sync_net.forward_lip(x)
        return x
