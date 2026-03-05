import os
import torch

from manifm.manifolds import SO3Power
from manifm.angles_utils import axis_angle_to_so3
from torch.utils.data.dataloader import default_collate


class AMASSDataset(torch.utils.data.Dataset):
    n_joints = 21
    dim = 9 * n_joints
    product_manifold = SO3Power(n_joints)

    def __init__(self, cfg, subset='train', sample_interval=100):

        assert subset in ['train', 'valid', 'test']
        self.root = cfg.pose_datadir
        self.subset = subset
        self.sample_interval = None if subset == "train" else sample_interval
        self.batch_size = cfg.optim.batch_size if subset == "train" else cfg.optim.val_batch_size
        self.num_workers = cfg.optim.n_workers

        self.body_poses = self.read_data()

        if self.sample_interval:
            self._sample(self.sample_interval)

        self.real_data_len = len(self.body_poses)

    def __getitem__(self, idx):
        body_pose = self.body_poses[idx % self.real_data_len]
        body_pose_so3 = axis_angle_to_so3(body_pose.view(-1, self.n_joints, 3))
        return body_pose_so3

    def __len__(self, ):
        return len(self.body_poses)

    def _sample(self, sample_interval):
        print(f'Class AMASSDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.body_poses = self.body_poses[::sample_interval]

    def read_data(self):
        data_path = os.path.join(self.root, self.subset)
        body_pose = torch.load(os.path.join(data_path, 'pose_body.pt'))
        return body_pose
    
    def get_loader(self, shuffle: bool = False, drop_last: bool = False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.pose_collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )

    def pose_collate_fn(self, batch):
        collated = default_collate(batch)
        return collated.view(-1, self.n_joints * 9)


def get_loaders(cfg):
    dataset = {}
    for mode in ["train", "valid", "test"]:
        dataset[mode] = AMASSDataset(cfg, mode)

    train_loader = dataset["train"].get_loader(shuffle=True, drop_last=True)
    val_loader = dataset["valid"].get_loader()
    test_loader = dataset["test"].get_loader()

    return train_loader, val_loader, test_loader


def get_product_manifold(cfg):
    n_joints = 21
    dim = 9 * n_joints
    product_manifold = SO3Power(n_joints)
    return product_manifold, dim
