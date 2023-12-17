import torch
from torch import nn
import numpy as np
from math import log
import torchvision.transforms as transforms
import torch.optim as optim
from apex import amp, optimizers
from data_loader.get_loader import get_loader, get_loader_label, get_loaderclu
from .utils import get_model_mme
from models.basenet import ResClassifier_MME


def get_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)

def get_clu_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loaderclu(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)



def get_dataloaders_label(source_data, target_data, target_data_label, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_label(source_data, target_data, target_data_label,
                            evaluation_data, data_transforms,
                            batch_size=conf.data.dataloader.batch_size,
                            return_id=True,
                            balanced=conf.data.dataloader.class_balance)

def get_models(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
#    device = kwargs["device_id"]
    G, dim = get_model_mme(net, num_class=num_class)

    C2 = ResClassifier_MME(num_classes=2*num_class,
                           norm=False, input_size=dim)
    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)
    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                      weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                       momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)

    [G, C1, C2], [opt_g, opt_c] = amp.initialize([G, C1, C2],
                                                  [opt_g, opt_c],
                                                  opt_level="O1")
    G = nn.DataParallel(G)
   # G = G
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])

    return G, C1, C2, opt_g, opt_c, param_lr_g, param_lr_c

def trans(tlabels):
    
    tlabels=torch.cat(tlabels,dim=0)
    tlabels=tlabels.cpu().tolist()
    tlabels=np.array(tlabels)

    return tlabels

def get_list(inputs,G,s,t):
    #s,t=get_clu_dataloaders(inputs)
    s_feature=[]
    slabels=[]
    s_index=[]
    #s_index1=[]
    #s_feature1=[]
    with torch.no_grad():
        #for iii, (imgs1,labels1,index1,_) in enumerate(s1):
         #   xxx=0
          #  s_feature1.append(imgs1)
           # s_index1.append(index1)
        for ii, (imgs,labels,index,_) in enumerate(s):
            feat=G(imgs)
            feat=feat.detach()
            slabels.append(labels)
            s_feature.append(feat)
            s_index.append(index)

    s_index=trans(s_index)
    slabels=trans(slabels)
    s_feature_list=trans(s_feature)
    s_feature=0

    #s_index_uniq = np.unique(s_index,return_index=True)
    #s_feature=s_feature_list[s_index_uniq[1]]
    torch.cuda.empty_cache()

    return s_feature_list,s_index

def get_thre(step,ratio,tao):
    thre = 0.1 + 0.1*(log(ratio)*(int(step/1500)/tao))
    return thre
