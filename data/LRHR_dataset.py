from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.dataroot = dataroot # Store dataroot as an attribute
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = None # Initialize env to None
            self.txn = None # Initialize txn to None
            # init the datalen - need to open env temporarily
            temp_env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            with temp_env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            temp_env.close() # Close the temporary env

            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            if self.env is None:
                # Open LMDB environment in the worker process
                self.env = lmdb.open(self.dataroot, readonly=True, lock=False,
                                     readahead=False, meminit=False)
                self.txn = self.env.begin(write=False)

            # Use the transaction opened above
            hr_img_bytes = self.txn.get(
                'hr_{}_{}'.format(
                    self.r_res, str(index).zfill(5)).encode('utf-8')
            )
            sr_img_bytes = self.txn.get(
                'sr_{}_{}_{}'.format(
                    self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
            )
            if self.need_LR:
                lr_img_bytes = self.txn.get(
                    'lr_{}_{}'.format(
                        self.l_res, str(index).zfill(5)).encode('utf-8')
                )
            # skip the invalid index
            while (hr_img_bytes is None) or (sr_img_bytes is None):
                new_index = random.randint(0, self.data_len-1)
                hr_img_bytes = self.txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(new_index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = self.txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = self.txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(new_index).zfill(5)).encode('utf-8')
                    )
            img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
            img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
