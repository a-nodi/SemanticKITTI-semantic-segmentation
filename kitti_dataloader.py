import os
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List
import MinkowskiEngine as ME
SEMANTIC_KITTI_PATH = "semantic_KITTI/dataset/"


class SemenaticKITTIDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data_path: str, sequence_set: List[int], data_type: str, learning_map: dict, voxel_size: float):
        super().__init__()
        self.data_path: str = data_path
        self.sequence_set: List[int] = sequence_set
        self.data_type: str = data_type
        assert len(sequence_set) >= 1, "no sequence has selected"
        self.current_sequence = sequence_set[0]
        self.learning_map = learning_map
        self.voxel_size = voxel_size
        self.list_of_scene_path, self.list_of_label_path = self.load_path()

    def load_path(self):
        list_of_scene_path, list_of_label_path = [], []
        for sequence in self.sequence_set:
            for scene_num in tqdm(range(len(os.listdir(os.path.join(self.data_path, 'sequences', '%02d' % sequence, 'velodyne'))))):
                scene_path = f"{os.path.join(self.data_path, 'sequences', '%02d' % sequence, 'velodyne', '%06d' % scene_num)}.bin"
                label_path = f"{os.path.join(self.data_path, 'sequences', '%02d' % sequence, 'labels', '%06d' % scene_num)}.label"

                list_of_scene_path.append(scene_path)
                list_of_label_path.append(label_path)

        return list_of_scene_path, list_of_label_path

    def load_one_scene(self, index):
        """_summary_

        Args:

        """

        pcd = np.fromfile(self.list_of_scene_path[index], dtype=np.float32)  # load point cloud from .bin
        pcd = pcd.reshape(-1, 4)  # reshape it to n x 4 matrix
        pcd = pcd[:, :3]  # remove intensity

        return pcd

    def load_one_label(self, index):
        """_summary_

        Args:

        """

        label = np.fromfile(self.list_of_label_path[index], dtype=np.uint32)  # load point cloud from .label
        label = label.reshape(-1, 1)  # reshape it to n x 1 matrix
        label = label & 0xFFFF  # filt out odd labels
        label = np.vectorize(self.learning_map.__getitem__)(label)  # remap labels

        return label

    def __getitem__(self, index):
        pcd = self.load_one_scene(index)
        feature = np.ones_like(pcd, dtype=np.float32)
        label = self.load_one_label(index)
        
        # Remove noises
        noise_index = np.where(label == 0)[0].tolist()
        pcd = np.delete(pcd, noise_index, axis=0)
        feature = np.delete(feature, noise_index, axis=0)
        label = np.delete(label, noise_index, axis=0)
        
        # del noise_index
        # shift label to -1
        label -= 1
        pcd /= self.voxel_size
        pcd, feature, label = ME.utils.sparse_quantize(pcd, feature, label, quantization_size=self.voxel_size)

        return pcd, feature, label

    def __len__(self):
        return len(self.list_of_scene_path)

def read_yaml(path):
    """
    
    """
    content = []
    with open(path, "r") as stream:
        content = yaml.safe_load(stream)

    return content

if __name__ == "__main__":
    
    content = read_yaml(os.path.join(SEMANTIC_KITTI_PATH, "semantic-kitti.yaml"))
    train_dataset = SemenaticKITTIDataset(
        SEMANTIC_KITTI_PATH,
        content['split']['train'],
        'train',
        content['learning_map'],
        voxel_size=0.05
        )

    for item in train_dataset:
        coords, feats, labels = item
        