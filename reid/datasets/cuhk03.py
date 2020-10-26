from ..utils.data import Dataset


class CUHK03(Dataset):
    def __init__(self, root, split_id=0, num_val=100):
        super(CUHK03, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Please follow README.md to prepare CUHK03 dataset.")

        self.load(num_val)
