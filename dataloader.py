from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import Dataset
import torch
import hdf5storage
import os

class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/"):
        self.data_views = []

        if db == "RGB-D":
            mat = hdf5storage.loadmat(os.path.join(path, 'RGB-D.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'CCV':
            mat = hdf5storage.loadmat(os.path.join(path, 'CCV.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'Cora':
            mat = hdf5storage.loadmat(os.path.join(path, 'Cora.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'Hdigit':
            mat = hdf5storage.loadmat(os.path.join(path, 'Hdigit.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.squeeze(mat['Y']).astype(np.int32)

        elif db == 'prokaryotic':
            # 1. 读取 mat 文件
            mat = hdf5storage.loadmat(os.path.join(path, 'prokaryotic.mat'))

            # 2. 动态识别所有视图字段（排除元数据 __*__ 和标签字段 truth）
            view_keys = [k for k in mat.keys()
                         if not k.startswith('__') and k != 'truth']
            if not view_keys:
                raise KeyError("prokaryotic.mat 中未找到任何视图字段，请检查字段名。")

            # 3. 按字段加载每个视图
            for key in view_keys:
                arr = mat[key]
                # 如果为稀疏格式，先转换 dense
                if hasattr(arr, 'toarray'):
                    arr = arr.toarray()
                arr = arr.astype(np.float32)
                # 保证样本在行上：若行数 < 列数，则转置
                if arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                self.data_views.append(arr)

            self.num_views = len(self.data_views)

            # 4. 加载标签 truth
            if 'truth' not in mat:
                raise KeyError("prokaryotic.mat 中缺少 'truth' 标签字段。")
            self.labels = np.squeeze(mat['truth']).astype(np.int32)

            # 5. 对每个视图做 Min–Max 归一化
            scaler = MinMaxScaler()
            for i in range(self.num_views):
                self.data_views[i] = scaler.fit_transform(self.data_views[i])

        else:
            raise NotImplementedError(f"Dataset '{db}' not supported.")

        # 最后：转为 Tensor 并移动到指定 device
        for i in range(self.num_views):
            self.data_views[i] = torch.from_numpy(self.data_views[i]).to(device)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        return sub_data_views, self.labels[index], index


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=False,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
