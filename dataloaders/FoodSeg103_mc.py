import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import xml.dom.minidom


class foodseg103(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # data_split = train / val
        self.root = root
        self.classnames = ["candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients"]
        
        
        self.data_split = data_split
        if data_split == 'trainval':
            self.labels_lab = np.load('/home/samyakr2/food/FoodSeg103/train_labels.npy', allow_pickle=True).item()
        
        if data_split == 'test':
            self.labels_lab = np.load('/home/samyakr2/food/FoodSeg103/test_labels.npy', allow_pickle=True).item()
            
        
        if annFile == "":
            self.annFile = os.path.join(self.root, 'Annotations')
        else:
            raise NotImplementedError

        image_list_file = os.path.join('/home/samyakr2/food/FoodSeg103', 'ImageSets', '%s.txt' % data_split)

        with open(image_list_file) as f:
            image_list = f.readlines()
        self.image_list = [a.strip() for a in image_list]

        
        if data_split == 'Train':
            num_examples = len(self.image_list)
            pick_example = int(num_examples * p)
            self.image_list = self.image_list[:pick_example]
        else:
            self.image_list = self.image_list

        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(img_size)
            transforms.Resize((img_size, img_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_transform = transforms.Compose([
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # seg_transform = transforms.Compose([
        #     # transforms.CenterCrop(img_size),
        #     transforms.Resize((img_size, img_size)),
        #     RandAugment(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])

        
        
        if self.data_split == 'trainval':
            self.transform = train_transform
            # self.seg_transform = seg_transform

        elif self.data_split == 'test':
            self.transform = test_transform
            # self.seg_transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Nus Wide' % self.data_split)

        # create the label mask
        self.mask = None
        self.partial = partial
        if data_split == 'trainval' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.image_list), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask], dim=1)
                torch.save(mask, os.path.join(self.root, 'Annotations', 'partial_label_%.2f.pt' % partial))
            else:
                mask = torch.load(os.path.join(self.root, 'Annotations', label_mask))
            self.mask = mask.long()

    def __len__(self):
        return len(self.image_list)

    def crop_image_into_eight(self, image):
        # image = Image.open(img_path).convert('RGB')
        width, height = image.size
        segments = [image.crop((i * width // 2, j * height // 4, (i + 1) * width // 2, (j + 1) * height // 4)) for j in range(4) for i in range(2)]
        return segments

    def __getitem__(self, index):
        
        if self.data_split == 'trainval':
                si = self.data_split[:-3]
        else:
            si = self.data_split
        
        img_path = os.path.join('/home/samyakr2/food/FoodSeg103', 'Images/img_dir/', si+'/', self.image_list[index])
        img = Image.open(img_path).convert('RGB')
        label_vector = self.labels_lab[self.image_list[index][:-4]]      
        targets = label_vector[1:].long()
        target = targets[None, ]
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked
        # print(img.shape)
        # breakpoint()
        segs = self.crop_image_into_eight(img)
        final_seg = []
        for seg in segs:
            t_seg = self.transform(seg)
            final_seg.append(t_seg)


        if self.transform is not None:
            img = self.transform(img)
        
        return (img, final_seg), target

    def name(self):
        return 'foodseg103'
