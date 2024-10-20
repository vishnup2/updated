import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import torchmetrics
import timm


from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class new_timm_model(nn.Module):
    def __init__(self, model, feature_cut):
        super().__init__()
        self.model = create_feature_extractor(model, return_nodes=feature_cut)  
        self.out_keys = list(feature_cut.values())
       

    def forward(self, x):
        out = self.model(x)
        return out[self.out_keys[0]], out[self.out_keys[1]]
       
       
num_classes = 80 # COCO dataset has 80 classes
#model and training details
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
feature_cut = {'blocks.11.attn.softmax':'final_attn', 'head':'output'}

model = new_timm_model(model, feature_cut)      


#### Toy Data --> Populate with the dataloading for Pascal images, labels, masks
imgs = torch.rand(5,3,224,224)
labels = torch.bernoulli(torch.rand(5,80))
adj_matrix = F.softmax(torch.rand((5,196,196)), dim=1) #apply softmax or sigmoid to binary adjacency matrix

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion1 = torch.nn.BCEWithLogitsLoss()
criterion2 = torch.nn.MSELoss()

epochs = 100
model.train()

for i in range(epochs):
    ###add batchloading for Pascal
    opt.zero_grad()
   
    attn_map, preds = model(imgs)
    attn_map = attn_map.mean(1).squeeze(1)[:,1:,1:] #average over heads and remove cls token
   
    loss1, loss2 = criterion1(preds, labels), criterion2(attn_map, adj_matrix)
    total_loss = loss1 + 0.4*loss2 #need to pla around with weighting for different dataset
   
    total_loss.backward()
    opt.step()
   
    print(f'BCE: {loss1.item()}, L2:{loss2.item()}') #gut check