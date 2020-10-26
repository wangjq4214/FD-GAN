from ..utils.data import Dataset


class DukeMTMC(Dataset):
    def __init__(self, root, split_id=0, num_val=100):
        super(DukeMTMC, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Please follow README.md to prepare DukeMTMC dataset.")

        self.load(num_val)
