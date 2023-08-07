import random
import torch

class DataSet():
    def __init__(self, data, input_len, output_len, val_date=10, test_date=10) -> None:
        self.data = data.tolist()
        self.input_len = input_len
        self.output_len = output_len
        self.val_date=val_date
        self.test_date=test_date

    def _get_max_sample_key(self):
        return len(self.data) - (self.input_len + self.output_len) \
                              - self.val_date \
                              - self.test_date
    
    def __getitem__(self, key):
        assert key <= len(self.data)-(self.input_len+self.output_len), f'Invalid Key {key}'
        x = torch.tensor(self.data[key:key+self.input_len]).float()
        y = torch.tensor(self.data[key+self.input_len:key+self.input_len+self.output_len]).float()
        return (x, y)

    def get_range_batch(self, begin, end):
        x, y = [], []
        for ix in torch.arange(begin, end):
            _x, _y = self[ix]
            x.append(_x)
            y.append(_y)

        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

    def get_val_batch(self, ):
        ceil = self._get_max_sample_key()+1
        val_date_begin, val_date_end = ceil, ceil+self.val_date

        return self.get_range_batch(val_date_begin, val_date_end)

    def get_test_batch(self, ):
        ceil = self._get_max_sample_key()+1+self.val_date
        test_date_begin, test_date_end = ceil, ceil+self.test_date

        return self.get_range_batch(test_date_begin, test_date_end)


def collate_fn(dataset:DataSet, batch_size):
    sample_ceil = dataset._get_max_sample_key()

    x, y = [], []
    for _ in range(batch_size):
        _x, _y = dataset[random.randint(0, sample_ceil)]
        x.append(_x)
        y.append(_y)
    x = torch.stack(x)
    y = torch.stack(y)

    return x, y