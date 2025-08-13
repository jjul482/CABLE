import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import torch
import pickle as pkl
import tarfile

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("../data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("../data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class ImageNet1K(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True  # store file paths

        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []

        self.class_order = np.arange(1000).tolist()  # ImageNet-1K has 1000 classes

    def _extract_archives(self, source_dir, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for fname in os.listdir(source_dir):
            if fname.endswith(".tar.gz"):
                tar_path = os.path.join(source_dir, fname)
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

    def _load_paths_and_labels(self, img_dir):
        """Return lists of image paths and integer labels."""
        img_paths = []
        labels = []
        synset_to_idx = {}
        next_idx = 0

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            synset = fname.split("_")[0]  # e.g., n01440764
            if synset not in synset_to_idx:
                synset_to_idx[synset] = next_idx
                next_idx += 1
            img_paths.append(os.path.join(img_dir, fname))
            labels.append(synset_to_idx[synset])

        return np.array(img_paths), np.array(labels)

    def download_data(self):
        root_dir = os.path.join(os.getcwd(), "data/imagenet-1k")
        extracted_dir = os.path.join(root_dir, "extracted")
        train_dir = os.path.join(extracted_dir, "train")
        test_dir = os.path.join(extracted_dir, "test")

        # Extract archives if needed
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print("Extracting ImageNet archives...")
            self._extract_archives(root_dir, extracted_dir)

            # Move extracted files into train/test folders
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            for fname in os.listdir(extracted_dir):
                if fname.startswith("train_images_"):
                    os.rename(os.path.join(extracted_dir, fname), train_dir)
                elif fname.startswith("test_images"):
                    os.rename(os.path.join(extracted_dir, fname), test_dir)

        # Load train/test paths and labels
        train_paths, train_labels = self._load_paths_and_labels(train_dir)
        test_paths, test_labels = self._load_paths_and_labels(test_dir)

        self.train_data = train_paths
        self.train_targets = train_labels
        self.test_data = test_paths
        self.test_targets = test_labels

    def get_loader(self, train=True, batch_size=64, num_workers=4):
        if train:
            subset_paths = self.train_data
            subset_labels = self.train_targets
            trsf = self.train_trsf
        else:
            subset_paths = self.test_data
            subset_labels = self.test_targets
            trsf = self.test_trsf

        dataset = PathDataset(
            subset_paths,
            subset_labels,
            transforms.Compose(trsf + self.common_trsf)
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class CORe50(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ]

    class_order = list(range(50))  # 50 objects (o1 to o50)

    def download_data(self):
        root = os.path.join(os.getcwd(), "data/core50")
        npz_path = os.path.join(root, "core50_imgs.npz")
        pkl_path = os.path.join(root, "paths.pkl")

        assert os.path.exists(npz_path), f"{npz_path} not found"
        assert os.path.exists(pkl_path), f"{pkl_path} not found"

        imgs = np.load(npz_path)["x"]  # shape: (164866, 128, 128, 3)
        with open(pkl_path, "rb") as f:
            paths = pkl.load(f)  # list of strings eg 's1/o12/img.png'

        assert len(paths) == len(imgs)

        # Extract class and session info from paths
        labels = []
        sessions = []
        for p in paths:
            session_str, object_dir = p.split("/")[:2]
            class_id = int(object_dir[1:]) - 1  # o1 → 0, o50 → 49
            session_id = int(session_str[1:])   # s1 → 1, s11 → 11
            labels.append(class_id)
            sessions.append(session_id)

        labels = np.array(labels)
        sessions = np.array(sessions)

        train_idx, test_idx = train_test_split(
            np.arange(len(imgs)),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        self.train_data = imgs[train_idx]
        self.train_targets = labels[train_idx]
        self.train_sessions = sessions[train_idx]

        self.test_data = imgs[test_idx]
        self.test_targets = labels[test_idx]
        self.test_sessions = sessions[test_idx]

    def get_loader(self, train=True, batch_size=64, num_workers=4):
        if train:
            dataset = CORe50NPZDataset(self.train_data, self.train_targets, self.train_trsf)
        else:
            dataset = CORe50NPZDataset(self.test_data, self.test_targets, self.test_trsf)

        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

class CORe50NPZDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        # if args["model_name"] == "coda_prompt":
        #     self.train_trsf = build_transform_coda_prompt(True, args)
        #     self.test_trsf = build_transform_coda_prompt(False, args)
        # else:
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/imagenet-r/train/"
        test_dir = "../data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/imagenet-a/train/"
        test_dir = "../data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True  # store image file paths instead of arrays

        # Transforms
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # Add shared transforms if needed
        ]

        self.class_order = np.arange(200).tolist()  # CUB has 200 classes

    def download_data(self, test_split=0.2):
        data_dir = os.path.join(os.getcwd(), "data/cub/")
        assert os.path.exists(data_dir), f"{data_dir} not found"

        # Load all images
        full_dataset = datasets.ImageFolder(data_dir)
        img_paths, labels = zip(*full_dataset.imgs)
        img_paths = np.array(img_paths)
        labels = np.array(labels)

        # Stratified split into train/test
        train_idx, test_idx = train_test_split(
            np.arange(len(img_paths)),
            test_size=test_split,
            stratify=labels,
            random_state=self.args.get("seed", 42)
        )

        self.train_data = img_paths[train_idx]
        self.train_targets = labels[train_idx]
        self.test_data = img_paths[test_idx]
        self.test_targets = labels[test_idx]

    def get_loader(self, train=True, batch_size=64, num_workers=4):
        # Use a subset of the split data
        if train:
            subset_paths = self.train_data
            subset_labels = self.train_targets
            trsf = self.train_trsf
        else:
            subset_paths = self.test_data
            subset_labels = self.test_targets
            trsf = self.test_trsf

        # Custom dataset to load images from stored paths
        dataset = PathDataset(
            subset_paths,
            subset_labels,
            transforms.Compose(trsf + self.common_trsf)
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/objectnet/train/"
        test_dir = "../data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/omnibenchmark/train/"
        test_dir = "../data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../data/vtab/train/"
        test_dir = "../data/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)