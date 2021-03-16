from typing import List, Dict, Optional
import os

import json
import tqdm
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset

from string import digits, ascii_uppercase
from zipfile import ZipFile

from support_utils import yaml_parser


ALPHABET_MASK = "0123456789ABEKMHOPCTYX"
SOURCE_CONF_PATH = 'conf/source_conf.yaml'


def download_url(url: str, path: str, unzip: bool = False) -> str:
    file_name = os.path.basename(url)
    if unzip:
        saving_path = os.path.join(path, file_name.split('.')[0])
    else:
        saving_path = os.path.join(path, file_name)

    if not os.path.exists(saving_path):
        print('Downloading from {} to {} ...'.format(url, path))
        os.system("wget -q {} {}".format(url, path))
        if unzip:
            zip_path = os.path.join(path, file_name)
            print('Unzip from {} to {} ...'.format(zip_path, path))
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(path)
            os.remove(zip_path)

    return saving_path


def compute_mask(text: str) -> Optional[str]:
    """Compute letter-digit mask of text.
    Accepts string of text.
    Returns string of the same length but with every letter replaced by 'L' and every digit replaced by 'D'.
    e.g. 'E506EC152' -> 'LDDDLLDDD'.
    Returns None if non-letter and non-digit character met in text.
    """
    mask = []
    for char in text:
        if char in digits:
            mask.append("D")
        elif char in ascii_uppercase:
            mask.append("L")
        else:
            return None
    return "".join(mask)


def check_in_alphabet(text: str, alphabet: str) -> bool:
    """Check if all chars in text come from alphabet.
    Accepts string of text and string of alphabet.
    Returns True if all chars in text are from alphabet and False else.
    """
    for char in text:
        if char not in alphabet:
            return False
    return True


def filter_data(config: List[Dict[str, str]], alphabet: str) -> List[Dict[str, str]]:
    """Filter config keeping only items with correct text.
    Accepts list of items.
    Returns new list.
    """
    config_filtered = []
    for item in tqdm.tqdm(config):
        text = item["text"]
        mask = compute_mask(text)
        if check_in_alphabet(text=text, alphabet=alphabet) and (mask in ["LDDDLLDD", "LDDDLLDDD"]):
            config_filtered.append({"file": item["file"],
                                    "text": item["text"]})
    return config_filtered


def get_prepared_data(
    path: str = '',
    alphabet: str = ALPHABET_MASK,
    conf_path: str = SOURCE_CONF_PATH,
    filtering: bool = True
) -> List[Dict[str, str]]:
    """
    1. Read urls for images and labels
    2. Download labels
    3. Download images, unzip, remove zip
    4. Add full path for images names
    5. Filter images by correct spelling of labels
    """
    source_conf = yaml_parser(path=conf_path)
    path = os.path.expanduser(path)

    url_images = source_conf['url_images']
    images_path = download_url(url=url_images, path=path, unzip=True)

    url_config_images = source_conf['url_config_images']
    config_path = download_url(url=url_config_images, path=path, unzip=False)
    with open(config_path, "rt") as fp:
        config = json.load(fp)

    config_full_paths = []
    for item in config:
        config_full_paths.append(
            {"file": os.path.join(images_path, item["file"]),
             "text": item["text"]})
    if filtering:
        config = filter_data(config=config_full_paths, alphabet=alphabet)
    else:
        config = config_full_paths

    return config


def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched images, sequences, and so.

    Текст номеров может иметь длину 8 (LDDDLLDD) или 9 (LDDDLLDDD).
    Класс DataLoader плохо справляется (из коробки) с данными переменного размера,
    поэтому будем использовать такую реализацию collate_fn.

    Можно посмотреть глазами, что все ок так (что в батчах будут номера только с одной маской):
    xs = [dataset[i] for i in range(4)]
    batch = collate_fn(xs)
    print(batch.keys())

    print("Image:", batch["image"].size())
    print("Seq:", batch["seq"].size())
    print("Seq:", batch["seq"])
    print("Seq_len:", batch["seq_len"])
    print("Text:", batch["text"])
    """
    images, seqs, seq_lens, texts = [], [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample["image"]).permute(2, 0, 1).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
    return batch


class RecognitionCarPlatesDataset(Dataset):
    """Class for training image-to-text mapping using CTC-Loss."""

    def __init__(self, config, alphabet=ALPHABET_MASK, transforms=None):
        """Constructor for class.
        Accepts:
        - config: list of items, each of which is a dict with keys "file" & "text".
        - alphabet: string of chars required for predicting.
        - transforms: transformation for items, should accept and return dict with keys
        "image", "seq", "seq_len" & "text".
        """
        super().__init__()
        self._config = config
        self.alphabet = alphabet
        self.image_names, self.texts = self._parse_root_()
        self._transforms = transforms

    def _parse_root_(self):
        image_names, texts = [], []
        for item in self._config:
            image_name = item["file"]
            text = item['text']
            texts.append(text)
            image_names.append(image_name)
        return image_names, texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Return dict with keys "image", "seq", "seq_len" & "text".
        Image is a numpy array, float32, [0, 1].
        Seq is list of integers.
        Seq_len is an integer.
        Text is a string.
        """
        image = cv2.imread(self.image_names[item]).astype(np.float32) / 255.
        text = self.texts[item]
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        if self._transforms is not None:
            output = self._transforms(output)
        return output

    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        Accepts string of text.
        Returns list of integers where each number is index of corresponding
        characted in alphabet + 1.
        """
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq


class Resize(object):
    """Class for resize transformation."""
    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        """Accepts item with keys "image", "seq", "seq_len", "text".
        Returns item with image resized to self.size.
        """
        item['image'] = cv2.resize(item['image'], self.size, interpolation=cv2.INTER_AREA)
        return item
