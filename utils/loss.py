from turtle import update
import torch
import faiss
import numpy as np
import torch.nn.functional as F

from eval import feat_get

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


def contrast_loss(feat,label,sfeat,slabels):
    loss = torch.tensor([0.]).cuda()
    sfeat = torch.from_numpy(sfeat.astype('float64')).cuda()
    feat_get = F.normalize(feat).mm(F.normalize(sfeat).t().clone().detach())
    temp_sims = feat_get.clone()
    feat_get /= 0.05
    slabels = torch.from_numpy(slabels).cuda()
    for i in range(len(feat)):
        associate_loss = 0
        ori_asso_ind = torch.nonzero(slabels == label[i]).squeeze(-1)
        if len(ori_asso_ind) == 0:
           pass
        else:
           temp_sims[i, ori_asso_ind] = -10000.0  # mask out positive
           sel_ind = torch.sort(temp_sims[i])[1][-20:]
           concated_input = torch.cat((feat_get[i, ori_asso_ind], feat_get[i, sel_ind]), dim=0)
           concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
           concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
           associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
           loss += 0.5 * associate_loss / len(feat_get)
    return loss

def contrast_loss1(feat,label,sfeat,slabels):
    loss = torch.tensor([0.]).cuda()
    sfeat = torch.from_numpy(sfeat.astype('float64')).cuda()
    feat_get = F.normalize(feat).mm(F.normalize(sfeat).t().clone().detach())
    temp_sims = feat_get.clone()
    feat_get /= 0.05
    slabels = torch.from_numpy(slabels).cuda()
    for i in range(len(feat)):
        associate_loss = 0
        ori_asso_ind = torch.nonzero(slabels == label[i]).squeeze(-1)
        if len(ori_asso_ind) == 0:
           pass
        else:
           temp_sims[i, ori_asso_ind] = -10000.0  # mask out positive
           sel_ind = torch.sort(temp_sims[i])[1][-5:]
           concated_input = torch.cat((feat_get[i, ori_asso_ind], feat_get[i, sel_ind]), dim=0)
           concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
           concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
           associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
           loss += 0.5 * associate_loss / len(feat_get)
    return loss


def loss_k( out1, out2, unk):
    #criterion = torch.nn.CrossEntropyLoss().cuda()
    mask = torch.zeros(len(out1)).cuda()
    mask += 1

    id = torch.nonzero(mask == 1).squeeze(-1)
    input_c1 = F.softmax(out1)
    #loss_kc1 = -(input_kc1*torch.log(input_kc1+1e-5)).sum()/len(out1)
    
    input_c2 = F.softmax(out2)
    loss_kc2 = -((input_c2*torch.log(input_c2+1e-5)).sum(1)).sum()/len(input_c2)

    loss_k =  loss_kc2
    #loss_unk = loss_c2

    return loss_k

def loss_c1( out1, out2, k):
    #criterion = torch.nn.CrossEntropyLoss().cuda()
    mask = torch.zeros(len(out1)).cuda()
    mask += 1
    #mask[unk]=0
    id = torch.nonzero(mask == 1).squeeze(-1)
    input_c1 = F.softmax(out1)
    #loss_kc1 = -(input_kc1*torch.log(input_kc1+1e-5)).sum()/len(out1)
    
    input_c2 = F.softmax(out2)
    loss_kc2 = -((input_c2[k]*torch.log(input_c1[k]+1e-5)).sum(1)).sum()/len(k)

    loss_k =  loss_kc2
    #loss_unk = loss_c2

    return loss_k

def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))
def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)


def loss_unk(out2,unk):
    input_c2 = F.softmax(out2)
    #loss_c2 = -((np.log(15)/2-input_c2[unk]*torch.log(input_c2[unk]+1e-5)).sum(1)).sum()/len(unk)

    loss_c2 = - (torch.log(input_c2[unk]).sum(1)-np.log(len(input_c2[0]))).sum()/(len(input_c2[0])*len(unk))
    #loss_c2 = - torch.log(input_c2[unk]).sum()/(len(input_c2[0])*len(unk)) #-np.log(len(input_c2[0]))
    return loss_c2

def loss_con_t(feat):

    sim = feat.mm(feat.t().clone().detach())
    denom = torch.norm(feat,dim=1).unsqueeze(1).mm(torch.norm(feat,dim=1).unsqueeze(1).t())
    cosm = sim/denom
    for i in range(len(cosm)):
        cosm[i,i] = 1.0
    #cosm_soft = torch.softmax(cosm,dim=1)
    ent = -cosm*(torch.log(cosm))
    loss = ent.sum()/len(ent)
    return loss


def loss_con(C2,feat,out_s,label):
    uqlabel = torch.unique(label)
    out_s = F.softmax(out_s)
    outtemp = out_s.cpu().clone()
    mask = torch.zeros_like(out_s).cuda()
    for i in range(len(feat)):
        outtemp[i][label[i]]=-1
        l = torch.argmax(outtemp[i])
        mask[i,l]+=1
        f_ind = torch.nonzero(label == label[i]).squeeze(-1)[0]
        mix = 0.3*feat[i]+0.7*feat[f_ind]
        if i == 0:
            mix_f = mix.unsqueeze(0)
        else:
            mix_f = torch.cat((mix_f,mix.unsqueeze(0)),dim=0)
    mix_out = C2(mix_f)
    mix_out = F.softmax(mix_out)
    loss = - (torch.log(mix_out).sum(1)-np.log(len(mix_out[0]))).sum()/(len(mix_out[0])*len(mix_out))
    #loss = - (mask*torch.log(mix_out)).sum()/len(mix_out)
    
    return loss




def loss_t( out1, out2,idx, unk, sel_labels):
    criterion = torch.nn.CrossEntropyLoss().cuda()

    unk_out1 = out1[unk]
    pred1 = torch.argmax(unk_out1,dim=1)
    unk_out2 = out2[unk]
    sel = torch.sort(unk_out2)[1][:,-2]
    
    pred_sel = torch.zeros_like(unk_out2).cuda()
    #sel_ind = torch.nonzero(sel_labels[idx]>=0).squeeze(-1)
    update_ind = torch.nonzero(sel_labels[idx][unk]<0).squeeze(-1)

    update_pred = pred1[update_ind]+1
    sel_labels[idx][update_ind] = update_pred.float()

    for i in range(len(unk)):
        temp=torch.zeros(len(out1[0])).cuda()
        id = idx.long()
        temp[(sel_labels[id][i]).long()] += 1
        pred_sel[i] = temp
    
    t_input = out2[unk]
    pred_sel = torch.argmax(pred_sel,dim=1).long()
    loss_unk = criterion(t_input,sel)

    mask = torch.zeros(len(out1)).cuda()
    mask += 1
    mask[unk] = 0
    
    input_kc1 = torch.softmax(out1,dim=1)
    loss_kc1 = -(input_kc1*torch.log(input_kc1+1e-5)).sum()/len(out1)

    input_c2 = torch.softmax(out2,dim=1)
    #loss_c2 = (1-((input_c2[unk]*torch.log(input_c2[unk]+1e-5)).sum(1)).sum())/len(unk)
    loss_kc2 = -(mask*(input_c2*torch.log(input_c2+1e-5)).sum(1)).sum()/len(out2)
   
    #loss_k =  loss_kc2
    #loss_unk = loss_c2

    return loss_kc1+loss_kc2,loss_unk


def open_entropy(out_open,unk_ind,pred,k_ind,other):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    #ent_open = torch.mean(torch.mean(torch.sum(input1, 1), 1))
    if len(other)==0:
        ent_open0=0
    else:
        out_open1=out_open[other].clone().cuda()
        ent_open0 = torch.mean(torch.mean(torch.sum(-out_open1 * torch.log(out_open1+ 1e-8), 1), 1)) #l_unc
    
    #if len(ind)!=0:
    #    ind = torch.from_numpy(ind)

    coef2=torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    coef2[k_ind, pred] = 1
   
    #coef2[ind]=0
    
    ent_open1 = torch.mean(torch.sum(-out_open[unk_ind, 0, :]*torch.log(out_open[unk_ind, 0, :]+ 1e-8), 1)) #l_unk
    ent_open2 = torch.mean(torch.sum(-out_open[:, 1, :]*torch.log(out_open[:, 1, :]+ 1e-8)*coef2, 1)) #l_k
    #ent_open2 = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]+ 1e-8)*coef2, 1))
    ent_open=(0.33*ent_open1+0.33*ent_open2+0.33*ent_open0)
    return ent_open