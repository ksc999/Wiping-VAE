from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_feat='train', transform=None):
        self.data_feat = data_feat
        self.transform = transform
        self.sample_pathes = self._load_all_image_pathes()
    
    def _load_all_image_pathes(self):
        filename = 'dataset_divide/' + self.data_feat + '.txt'   
        file = open(filename)
        sample_pathes = []
        for file_path in file.readlines():
            sample_pathes.append(file_path[:-1])
        return sample_pathes

    def _load_one_image(self, img_path):
        return Image.open(img_path).convert('RGB')       

    def __getitem__(self, index):
        img_path = self.sample_pathes[index]
        img = self._load_one_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.sample_pathes)


if __name__ == '__main__':
    test_dataset = MyDataset()
    print(test_dataset[0])