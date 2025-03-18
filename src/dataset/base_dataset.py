class BaseDataset:
    def __init__(self):
        pass

    def get_graph_pair(self, cls, shuffle):
        raise NotImplementedError
