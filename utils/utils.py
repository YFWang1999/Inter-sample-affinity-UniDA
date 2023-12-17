from models.basenet import *
import os
import torch
import neptune
import socket


def get_model_mme(net, num_class=13, temp=0.05, top=False, norm=True):
    dim = 2048
    if "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    if top:
        dim = 1000
    print("selected network %s"%net)
   # from ipdb import set_trace
   # set_trace()
    return model_g, dim

def log_set(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    network = kwargs["network"]
    conf_file = kwargs["config_file"]
    script_name = kwargs["script_name"]
    multi = kwargs["multi"]
    #args = kwargs["args"]

    target_data = os.path.splitext(os.path.basename(target_data))[0]
    logname = "{file}_{source}2{target}_{network}_hp_{hp}".format(file=script_name.replace(".py", ""),
                                                                               source=source_data.split("_")[1],
                                                                               target=target_data,
                                                                               network=network,
                                                                               hp=str(multi))
    logname = os.path.join("record", kwargs["exp_name"],
                           os.path.basename(conf_file).replace(".yaml", ""), logname)
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))
    print("record in %s " % logname)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logname, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("{}_2_{}".format(source_data, target_data))
    return logname


def save_model(model_g, model_c1, model_c2, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c1_state_dict': model_c1.state_dict(),
        'c2_state_dict': model_c2.state_dict(),
    }
    torch.save(save_dic, save_path)

def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c

def lo(k_ent,unk_ent):
    with open("123ent.txt",'a') as f:
        f.write('k_ent: '+str(k_ent)+'        unk_ent:'+str(unk_ent))
        f.write('\n')

def oh(num,num_unk_gt,acc,step):
    with open("122.txt",'a') as f:
        f.write('step:' + str(step)+'num: '+str(num)+'/'+str(num_unk_gt)+'   acc'+str(acc))
        f.write('\n')

def unkova(acc,step):
    with open("unk.txt",'a') as f:

        f.write('step:' + str(step)+'   acc'+str(acc))
        f.write('\n')


def lcon(acc1,acc2,unk,p,gt,step):
    with open("recd_oh.txt",'a') as f:
        f.write('step:' + str(step) +'   k_acc:'+str(acc1) +'   num:'+str(len(p)) +'   unk_acc:'+str(acc2) + '   pred:'+str(p) + '   gt:'+str(gt))
        f.write('\n')

def lunk(acc2,unk,step,ip_unk,ip_k):
    with open("ip_oh.txt",'a') as f:
        f.write('step:' + str(step)   +'   unk_acc:'+str(acc2) + '   unk_idx:'+str(unk)+'   ip_unk:'+str(ip_unk)+'   ip_k:'+str(ip_k))
        f.write('\n')
        
def lscore(step,ls1_unk,ls1_k,ls2_unk,ls2_k):
    with open("lscore_oh.txt",'a') as f:
        f.write('step:' + str(step)+'   ls1_unk:'+str(ls1_unk)+'   ls1_k:'+str(ls1_k)+'   ls2_unk:'+str(ls2_unk)+'   ls2_k:'+str(ls2_k))
        f.write('\n')

def rec(con_recd, labels):
    with open("con_recd.txt",'a') as f:
        f.write('Confidence:  ' + str(con_recd)+'\n'+'labels:  '+str(labels))
        f.write('\n')


def rec1(con_recd, labels):
    with open("con_recd_o_ori.txt",'a') as f:
        f.write('Confidence:  ' + str(con_recd)+'\n'+'labels:  '+str(labels))
        f.write('\n')
        
def rec2(kdis, labels):
    with open("kdis.txt",'a') as f:
        f.write('Confidence:  ' + str(kdis)+'\n'+'labels:  '+str(labels))
        f.write('\n')

def rec3(kdis, labels):
    with open("kdis_o_ori.txt",'a') as f:
        f.write('Confidence:  ' + str(kdis)+'\n'+'labels:  '+str(labels))
        f.write('\n')



def recd_txt(kdis, con,step, txt):
    with open(txt,'a') as f:
        f.write('step:  '+str(step)+'  ENTunknown: ' + str(kdis)  +'  ENTknown: '+str(con))
        f.write('\n')


def recd_txt1(kdis, con,step, txt):
    with open(txt,'a') as f:
        f.write('step:  '+str(step)+'  ENTunknown: ' + str(kdis)  +'  ENTknown: '+str(con))
        f.write('\n')