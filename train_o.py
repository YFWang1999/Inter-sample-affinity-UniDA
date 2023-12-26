from __future__ import print_function
import yaml
import easydict
import os
import torch
import faiss
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import log_set, save_model,lo,simjj,computesim
from utils.lossde import source_loss,open_entropy, open_entropy1
from utils.k_means import KMeans
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models, get_thre, get_list, get_loaderclu
from eval import test
import argparse

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml',
                    help='/path/to/config/file')

parser.add_argument('--source_data', type=str,
                    default='./utils/source_list.txt',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./utils/target_list.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='office_wopt',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=True, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/ova_model",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.1,
                    help='weight factor for adaptation')
parser.add_argument('--lam', type=float,
                    default=1)
parser.add_argument('--nn', type=float,
                    default=5)
args = parser.parse_args()

config_file = args.config
conf = yaml.safe_load(open(config_file))
save_config = yaml.safe_load(open(config_file))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()
lam=args.lam

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

source_loader, target_loader, \
test_loader, target_folder = get_dataloaders(inputs)

logname = log_set(inputs)

G, C1, C2, opt_g, opt_c, \
param_lr_g, param_lr_c = get_models(inputs)
ndata = target_folder.__len__()
lens = len(source_loader.dataset.labels)
lent = ndata
ratio = lens/lent
tao = 20



def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)

        txx = get_thre(step,ratio,tao)

        if step  == 0 :
            print('Building memory...')
            sfeatures,s_index,slabels=get_list(G,source_loader)
            sfeatures=sfeatures.astype('float32')
            alltfeatures,t_index,tlabels=get_list(G,target_loader)
            alltfeatures=alltfeatures.astype('float32')

            print('Complete!')


        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        C2.module.weight_norm()

        ## Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)
        out_open = C2(feat)
        ## source classification loss

        if step>=0:
            feat_n=feat.clone().cpu().detach().numpy()
            sfeatures[data_s[2]]=0.9*sfeatures[data_s[2]]+0.1*feat_n
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        out_s=F.softmax(out_open,1)
        out1_norm=out_s[:,1,:].clone()
        for j in range(len(out_s)):
            out1_norm[j]=out1_norm[j]/out1_norm[j].sum()
        max_mean=(out1_norm.max(1)[0]).mean()
        num=100
        
        open_loss_pos, open_loss_neg = source_loss(out_open, label_s)
        ## b x 2 x C
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all =  loss_open
        ind = data_s[2].cpu().detach().numpy()
        tempfeat = feat.clone().detach().cpu().numpy()
        tempfeat = tempfeat.astype('float32')
        for k in range(len(feat)):
            idxx = np.nonzero(s_index==ind[k])[0]
            sfeatures[idxx]=0.9*sfeatures[idxx]+0.1*tempfeat[k]
        log_string = 'Train {}/{} \t ' \
                     'Loss Source: {:.4f} ' \
                     'Loss Open: {:.4f} ' \
                     'Loss Open Source Positive: {:.4f} ' \
                     'Loss Open Source Negative: {:.4f} '
        log_values = [step, conf.train.min_step,
                      loss_open.item(),
                      open_loss_pos.item(), open_loss_neg.item()]
        if not args.no_adapt:
            feat_t = G(img_t)
            out_open_t = C2(feat_t)
            unk_pred=[]
            
            out_open_t = out_open_t.view(img_t.size(0), 2, -1)
            if step>=500:
                tfeatures = feat_t.detach().cpu().numpy()
                tfeatures = tfeatures.astype('float32')
                features=np.concatenate((sfeatures,tfeatures),axis=0).astype('float32')
                n, dim = features.shape[0], features.shape[1]
                index = faiss.IndexFlatL2(dim)
               
                index.add(sfeatures)
                distances, indices = index.search(tfeatures, 5+int(args.nn)) # Sample itself is included
                
                index1 = faiss.IndexFlatL2(dim)
                index1.add(alltfeatures)
                distances,indeices1 = index1.search(alltfeatures,6+int(args.nn))
                k_pred_ori=[]
                unk_pred_ori=[]
                jj_rec=[]
                label_rec=[]
                for ii in range(len(indices)):
                    plabel = np.argmax(np.bincount(slabels[indices][ii]))
                    if np.max(np.bincount(slabels[indices][ii])) <=3:
                        unk_pred_ori.append(ii)
                        jj_rec.append(-1)
                    else:
                        ax=sfeatures[indices[ii]][np.nonzero(slabels[indices[ii]] == plabel )[0]]
                        ax1=alltfeatures[indeices1[ii][1:(len(ax)+1)]]
                        eigen_vals1, eigen_vecs1 = np.linalg.eig(np.cov(ax))
                        eigen_vals2, eigen_vecs2 = np.linalg.eig(np.cov(ax1))
                        jj = np.dot(eigen_vecs1[0],eigen_vecs2[0])/(np.linalg.norm(eigen_vecs1[0])*np.linalg.norm(eigen_vecs2[0]))
                        jj_rec.append(jj)
                        label_rec.append(data_t[1][ii].tolist())
                        if jj>=0.1:
                            k_pred_ori.append(ii)
                        else:
                            unk_pred_ori.append(ii)
                    
                k_pred_ori = torch.tensor(k_pred_ori)


                confidence=[]
                for i in range(len(k_pred_ori)):
                    sss=sfeatures [indices[k_pred_ori[i]]]
                    sss=sss.astype('float64')
                    sss=torch.from_numpy(sss)
                    out=C2(sss)
                    out=out.view(sss.size(0),2,-1)
                    out=F.softmax(out,1)
                    acc_mean=out[:,1,:].mean(0)
                    acc_mean=acc_mean/acc_mean.sum()
                    confidence.append(acc_mean.unsqueeze(0))
                con=torch.cat(confidence,dim=0)
                con_max=con.max(1)[0]
                plabels=con.max(1)[1]
                
                unk_pred=torch.nonzero(con_max<max_mean*(0.8)).squeeze(-1)
                unk_pred_ori = torch.tensor(unk_pred_ori).cuda()
                if len(unk_pred_ori)!=0:
                    unk_pred = torch.cat((unk_pred,unk_pred_ori))
                
                k_pred=torch.nonzero(con_max>=max_mean).squeeze(-1)
                pred=plabels[k_pred]
                temp1=set(unk_pred.tolist())
                temp2=set(k_pred.tolist())
                temp3=set([i for i in range(36)])
                other=torch.tensor(list(temp3.difference(temp1.union(temp2)))).cuda()


            
            if step<500:
                ent_open=open_entropy1(out_open_t)
            else:
                #other=[]
                ent_open = open_entropy(out_open_t,unk_pred,pred,k_pred,other,lam)
            
            all += args.multi * ent_open
            log_values.append(ent_open.item())
            log_string += "Loss Open Target: {:.6f}"
        with amp.scale_loss(all, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))
        #if step > 0 and step % conf.test.test_interval == 0:
        if step > 0 and step % 500 == 0:
            #dict1 = unpickle('xxx/cifar-10-batches-py/data_batch_1')
            acc_o, h_score = test(step, test_loader, logname, n_share, G,
                                  [C1, C2], open=open)
            print("acc all %s h_score %s " % (acc_o, h_score))
            G.train()
            C1.train()

            if step == 60000 and args.save_model:
                save_path = "%s_%s.pth"%(args.save_path, step)
                save_model(G, C1, C2, save_path)


train()
