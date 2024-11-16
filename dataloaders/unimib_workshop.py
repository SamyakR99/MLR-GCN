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


class unimib(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", label_mask=None, partial=1+1e-6):
        # data_split = train / val
        self.root = root
        self.classnames = ['arancia', 'arrosto', 'arrosto_di_vitello', 'banane', 'bruscitt', 'budino', 'carote', 'cavolfiore', 'cibo_bianco_non_identificato', 'cotoletta', 'crema_zucca_e_fagioli', 'fagiolini', 'finocchi_gratinati', 'finocchi_in_umido', 'focaccia_bianca', 'guazzetto_di_calamari', 'insalata_2_(uova mais)', 'insalata_mista', 'lasagna_alla_bolognese', 'mandarini', 'medaglioni_di_carne', 'mele', 'merluzzo_alle_olive', 'minestra', 'minestra_lombarda', 'orecchiette_(ragu)', 'pane', 'passato_alla_piemontese', 'pasta_bianco', 'pasta_cozze_e_vongole', 'pasta_e_ceci', 'pasta_e_fagioli', 'pasta_mare_e_monti', 'pasta_pancetta_e_zucchine', 'pasta_pesto_besciamella_e_cornetti', 'pasta_ricotta_e_salsiccia', 'pasta_sugo', 'pasta_sugo_pesce', 'pasta_sugo_vegetariano', 'pasta_tonno', 'pasta_tonno_e_piselli', 'pasta_zafferano_e_piselli', 'patate/pure', 'patate/pure_prosciutto', 'patatine_fritte', 'pere', 'pesce_(filetto)', 'pesce_2_(filetto)', 'piselli', 'pizza', 'pizzoccheri', 'polpette_di_carne', 'riso_bianco', 'riso_sugo', 'roastbeef', 'rosbeef', 'rucola', 'salmone_(da_menu_sembra_spada_in_realta)', 'scaloppine', 'spinaci', 'stinco_di_maiale', 'strudel', 'torta_ananas', 'torta_cioccolato_e_pere', 'torta_crema', 'torta_crema_2', 'torta_salata_(alla_valdostana)', 'torta_salata_3', 'torta_salata_rustica_(zucchine)', 'torta_salata_spinaci_e_ricotta', 'yogurt', 'zucchine_impanate', 'zucchine_umido']

        
        self.data_split = data_split
        self.labels_lab = np.load('/home/samyakr2/food_datasets/UNIMIB2016/unimib_labels.npy', allow_pickle=True).item()
        
        if annFile == "":
            self.annFile = os.path.join(self.root, 'Annotations')
        else:
            raise NotImplementedError

        image_list_file = os.path.join('/home/samyakr2/food_datasets/UNIMIB2016/',  '%s.txt' % data_split)

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

        if self.data_split == 'train':
            self.transform = train_transform
        elif self.data_split == 'test':
            self.transform = test_transform
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

    def __getitem__(self, index):
        
        img_path = os.path.join('/home/samyakr2/food_datasets/UNIMIB2016/images/original/', self.image_list[index])
        img = Image.open(img_path).convert('RGB')
        label_vector = torch.tensor(self.labels_lab[self.image_list[index][:-4]])
        targets = label_vector.long()
        target = targets[None, ]
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def name(self):
        return 'unimib'
