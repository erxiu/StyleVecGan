import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from utils.util import mu_law_encode, mu_law_decode

class CustomerDataset(Dataset):
    def __init__(self,
                 path,
                 upsample_factor=13500,
                 local_condition=True):

        self.path = path
        self.metadata = self.get_metadata(path)

        self.local_condition = local_condition
        self.upsample_factor = upsample_factor

    def __getitem__(self, index):

        try:
            sample = np.load(os.path.join(self.path, 'audio', self.metadata[index]))
            condition = np.load(os.path.join(self.path, 'vec', self.metadata[index]))
        except:
            return self.__getitem__(index+1)

        length = min([len(sample), (len(condition) - 2) * self.upsample_factor])

        sample = sample[: length]
        condition = condition[1: length // self.upsample_factor + 1, :] # condition[0, :] 是 [CLS] 向量，所以要去掉

        sample = sample.reshape(-1, 1)

        if self.local_condition:
            return sample, condition
        else:
            return sample

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self, path):
        with open(os.path.join(path, 'names.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        return metadata

class CustomerCollate(object):

    def __init__(self, upsample_factor=13500, char_num=5):
        self.upsample_factor = upsample_factor
        self.char_num = char_num

    def __call__(self, batch):
        return self._collate_fn(batch)

    def _pad(self, x, max_len):
        x = np.pad(x, [[0, max_len - len(x)], [0, 0]], 'constant')
        return x

    def _collate_fn(self, batch):

        sample_batch = []
        condition_batch = []

        for sample, condition in batch:
            if len(condition) < self.char_num:
                sample = self._pad(sample, self.char_num*self.upsample_factor)
                condition = self._pad(condition, self.char_num)

            elif len(condition) > self.char_num:
                # lc_index = np.random.randint(0, len(condition) - self.char_num)
                lc_index = 0

                sample = sample[lc_index*self.upsample_factor:(lc_index+self.char_num)*self.upsample_factor]
                condition = condition[lc_index:(lc_index+self.char_num)]
            else:
                pass

            sample_batch.append(sample)
            condition_batch.append(condition)


        sample_batch = np.stack(sample_batch)
        condition_batch = np.stack(condition_batch)

        sample_batch = mu_law_encode(sample_batch)
        sample_batch = mu_law_decode(sample_batch)

        samples = torch.FloatTensor(sample_batch).transpose(1, 2)
        conditions = torch.FloatTensor(condition_batch).transpose(1, 2)

        return samples, conditions


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = CustomerDataset("data/train", local_condition=True)
    collate = CustomerCollate()

    train_data_loader = DataLoader( train_dataset,
                                    collate_fn=collate,
                                    batch_size=4,
                                    num_workers=1,
                                    shuffle=True,
                                    pin_memory=True)

    for sample, condition in train_data_loader:
        print(sample.shape, condition.shape)
