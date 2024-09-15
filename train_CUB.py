import argparse
import os
import random
import sys
import time
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import classifier
import model
import util
import torch
import numpy as np
import wandb
from sklearn.metrics.pairwise import cosine_similarity
import pre_cls_seen as pre_cls
import classifiervis as classifier_zero
import os
import loss
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

wandb.init(project='VADS', entity='hwj',config='./train_clswgan_cub_prompt_sent_vit.yaml')
opt = wandb.config
print(opt)
contrastive_loss = loss.Contrastive_loss_clear(0.05)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

data = util.DATA_LOADER(opt)
print("Training Samples: ", data.ntrain)

pretrain_cls = pre_cls.CLASSIFIER(data, data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 4096,
                                     opt.pretrain_classifier)

for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False
pretrain_cls.model.eval()




netG = model.Generator(opt)
netD = model.Discriminator(opt)
netE = model.Encoder_Prompt(opt)
netPrompt = model.PromptsLearner(opt)
netATT = model.Att_update(opt)
netATT.load_state_dict(torch.load('/home/hwj/CVPR_CODE/data/CUB/att_up_model_CUB.pth'))
cls_criterion = nn.NLLLoss()
criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)
att= torch.LongTensor(opt.nclass_all,opt.attSize)
beta=0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netE.cuda()
    netPrompt.cuda()
    netATT.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()
    #batch_label = batch_label.cuda()
    att = att.cuda()
    
def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    #BCE = (recon_x - x).pow(2).sum(1).mean()
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    #return (KLD)
    #return (BCE)
    return (BCE + KLD)


def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))
    batch_label.copy_(batch_label)
    return batch_label

# def generate_syn_feature_vis(generator,classes, attribute,num,netVis=None):
#     nclass = classes.size(0)
#     syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
#     syn_label = torch.LongTensor(nclass*num) 
#     syn_att = torch.FloatTensor(num, opt.attSize)
#     syn_noise = torch.FloatTensor(num, opt.nz)
#     if opt.cuda:
#         syn_att = syn_att.cuda()
#         syn_noise = syn_noise.cuda()
#     for i in range(nclass):
#         iclass = classes[i]
#         iclass_att = attribute[iclass]
#         syn_att.copy_(iclass_att.repeat(num, 1))
#         syn_noise.normal_(0, 1)
#         with torch.no_grad():
#             syn_noisev = Variable(syn_noise)
#             syn_attv = Variable(syn_att)
#         fake = generator(syn_noisev,c=syn_attv)
#         output = fake
#         syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
#         syn_label.narrow(0, i*num, num).fill_(iclass)

#     return syn_feature, syn_label
def generate_syn_feature_prompt(generator,classes, attribute,num,netPrompt=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        #prompt = netPrompt().expand(num,-1)
        #syn_noisev = prompt+ syn_noisev
        #c = torch.cat([prompt,syn_attv],dim=-1)
        prompt,_,_ = netPrompt(z=syn_noisev)
        #fake = generator(syn_noisev,c=syn_attv)
        #fake = generator(prompt,c=syn_attv)
        fake = generator(0.5*prompt+0.5*syn_noisev,c=syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# def generate_syn_feature_prompt_from_ctx(generator,classes, attribute,num,netPrompt=None,u_s_sortid=u_s_sortid,seen_ctx = None):
#     nclass = classes.size(0)
#     syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
#     syn_label = torch.LongTensor(nclass*num) 
#     syn_att = torch.FloatTensor(num, opt.attSize)
#     syn_noise = torch.FloatTensor(num, opt.nz)
#     if opt.cuda:
#         syn_att = syn_att.cuda()
#         syn_noise = syn_noise.cuda()
#     for i in range(nclass):
#         iclass = classes[i]
#         unseen_id = u_s_sortid[i]
#         iclass_att = attribute[iclass]
#         syn_att.copy_(iclass_att.repeat(num, 1))
#         syn_noise.normal_(0, 1)
#         with torch.no_grad():
#             syn_noisev = Variable(syn_noise)
#             syn_attv = Variable(syn_att)
#         k = 0
#         for j in unseen_id:
#             k += 1
#             unseen_ctx = seen_ctx[j]
#         unseen_ctx = unseen_ctx / k
#         unseen_ctx = unseen_ctx.unsqueeze(0).expand(num,-1)


#         prompt = netPrompt(z=syn_noisev,unseen_ctx = unseen_ctx)
#         fake = generator(prompt,c=syn_attv)
#         output = fake
#         syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
#         syn_label.narrow(0, i*num, num).fill_(iclass)

#     return syn_feature, syn_label

def generate_syn_feature(generator,classes, attribute,num,netFR=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        fake = generator(syn_noisev,c=syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def generate_syn_feature_with_grad(netG, classes, attribute, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = netG(Variable(syn_noise), Variable(syn_att))
    return syn_feature, syn_label.cpu()


optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerATT         = optim.Adam(netATT.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

optimizerE = optim.Adam(netE.parameters(),lr = 1e-3)
optimizerP = optim.Adam(netPrompt.parameters(),lr = 1e-3)

sigmoid = nn.Sigmoid()

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

best_gzsl_acc = 0
best_zsl_acc = 0
att=data.attribute.cuda()
for epoch in range(opt.nepoch):
    if epoch<opt.update_epoch:
        for i in range(0, data.ntrain, opt.batch_size):
            for p in netD.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = False
            


            for iter_d in range(opt.critic_iter):
                batch_label=sample()
                batch_label = batch_label.cuda()
                netD.zero_grad()
                netE.zero_grad()
                netPrompt.zero_grad()
                # train with realG
                input_resv = Variable(input_res)
            
                input_attv_ori = Variable(input_att)
                input_attv = Variable(att[batch_label])
                

                criticD_real = netD(input_resv, input_attv_ori)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone)

            
                mean,log_var,con_fea,_ = netE(input_resv)
                std = torch.exp(0.5*log_var)
                z = torch.randn(mean.shape).cuda()
                z  = std * z + mean
                con_loss = contrastive_loss(con_fea,con_fea,input_label)

                
                prompt,_,_ = netPrompt(z=z)
            
            
                noise.normal_(0, 1)
                noisev = Variable(noise)
        
                fake = netG(0.5*prompt+0.5*noisev, c = input_attv_ori)
                #vae_loss = loss_fn(fake,input_resv,mean,log_var)
                
                criticD_fake = netD(fake.detach(), input_attv_ori)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one)

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_attv)+opt.con_loss*con_loss#+0.1*prompt_loss
                
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
            



        
            for p in netG.parameters():
                    p.requires_grad = True
            for p in netD.parameters():
                    p.requires_grad = False

            netG.zero_grad()
            input_attv_ori = Variable(input_att)
            input_attv = Variable(att[batch_label]) 
            
            mean,log_var,con_fea,_ = netE(input_resv)
            std = torch.exp(0.5*log_var)
            z = torch.randn(mean.shape).cuda()
            z  = std * z + mean
            con_loss = contrastive_loss(con_fea,con_fea,input_label)


            prompt,m,v = netPrompt(z=z)
            prompt_loss =  -0.5 * torch.sum(1 + v - m.pow(2) - v.exp())/ m.size(0)

            noise.normal_(0, 1)
            noisev = Variable(noise)
        
            recon_x = netG(0.5*prompt+0.5*noisev, c = input_attv_ori)
        
            fake = recon_x
            criticG_fake = netD(recon_x, input_attv_ori)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            vae_loss = loss_fn(recon_x,input_resv,mean,log_var)
        

            c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

            errG = opt.gammaG*G_cost + opt.vae_loss*vae_loss+opt.cls_weight * c_errG+opt.con_loss*con_loss +0.5* prompt_loss#+con_loss#+ opt.prompt_loss * prompt_loss+

            errG.backward()
            optimizerE.step()
            optimizerP.step()
            
            optimizerG.step()
    else:
        for i in range(0, data.ntrain, opt.batch_size):
            for p in netD.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = False
            


            for iter_d in range(opt.critic_iter):
                batch_label=sample()
                batch_label = batch_label.cuda()
                netD.zero_grad()
                netE.zero_grad()
                netPrompt.zero_grad()
                netATT.zero_grad()
                # train with realG
                input_resv = Variable(input_res)
            
                input_attv_ori = Variable(input_att)
                input_attv = Variable(att[batch_label])
                
                update_att = netATT(att)[batch_label]
                criticD_real = netD(input_resv, update_att)
                criticD_real = criticD_real.mean()
                criticD_real.backward(mone)

            
                mean,log_var,con_fea,_ = netE(input_resv)
                std = torch.exp(0.5*log_var)
                z = torch.randn(mean.shape).cuda()
                z  = std * z + mean
                con_loss = contrastive_loss(con_fea,con_fea,input_label)

                
                prompt,_,_ = netPrompt(z=z)
            
            
                noise.normal_(0, 1)
                noisev = Variable(noise)
        
                fake = netG(0.5*prompt+0.5*noisev, c = update_att)
                #vae_loss = loss_fn(fake,input_resv,mean,log_var)
                update_att = netATT(att)[batch_label]
                criticD_fake = netD(fake.detach(), update_att)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(one,retain_graph=True)
                att_loss = F.l1_loss(update_att,input_attv_ori)
                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, update_att)+opt.att_loss*att_loss+opt.con_loss*con_loss#+0.1*prompt_loss0.5

               
                
                gradient_penalty.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                optimizerD.step()
                optimizerATT.step()
            



        
            for p in netG.parameters():
                    p.requires_grad = True
            for p in netD.parameters():
                    p.requires_grad = False

            netG.zero_grad()
            input_attv_ori = Variable(input_att)
            input_attv = Variable(att[batch_label]) 
            
            mean,log_var,con_fea,_ = netE(input_resv)
            std = torch.exp(0.5*log_var)
            z = torch.randn(mean.shape).cuda()
            z  = std * z + mean
            con_loss = contrastive_loss(con_fea,con_fea,input_label)


            prompt,m,v = netPrompt(z=z)
            prompt_loss =  -0.5 * torch.sum(1 + v - m.pow(2) - v.exp())/ m.size(0)

            noise.normal_(0, 1)
            noisev = Variable(noise)
            
            update_att = netATT(att)[batch_label]


            recon_x = netG(0.5*prompt+0.5*noisev, c = update_att)
        
            fake = recon_x
            criticG_fake = netD(recon_x, update_att)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            vae_loss = loss_fn(recon_x,input_resv,mean,log_var)
        

            c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

            errG = opt.gammaG*G_cost + opt.vae_loss*vae_loss+opt.cls_weight * c_errG+opt.con_loss*con_loss +0.5* prompt_loss#+con_loss#+ opt.prompt_loss * prompt_loss+

            errG.backward()
            optimizerE.step()
            optimizerP.step()
            
            optimizerG.step()

            
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f'% (epoch, opt.nepoch, D_cost.item(), G_cost.item()))#,end=" ")
    netG.eval()
    netPrompt.eval()
    if epoch < opt.update_epoch:
        syn_feature, syn_label = generate_syn_feature_prompt(netG,data.unseenclasses, data.attribute, opt.syn_num,netPrompt=netPrompt)
    else:
        update_att_new = netATT.update_att()
        syn_feature, syn_label = generate_syn_feature_prompt(netG,data.unseenclasses, update_att_new, opt.syn_num,netPrompt=netPrompt)
    ### Concatenate real seen features with synthesized unseen features
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = opt.nclass_all

    # _,_,X = netE(train_X.cuda())
    # train_X = torch.cat((train_X.cpu(),X.cpu()),dim=1)
    # _,_,X_syn = netE(syn_feature.cuda())
    # syn_feature = torch.cat((syn_feature.cpu(),X_syn.cpu()),dim=1)

    if opt.gzsl:  
        if opt.final_classifier == 'softmax':
            ### Train GZSL classifier
            gzsl_cls = classifier_zero.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, generalized=True, final_classifier=opt.final_classifier, netVis=netE,netATT=None, dec_size=opt.attSize, dec_hidden_size=(opt.latensize*2),_att = att.cpu())
            if best_gzsl_acc <= gzsl_cls.H:
                best_gzsl_epoch= epoch
                best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
                # path_G = f'./saved_model/netG_{opt.dataset}_VADS.pth'
                # torch.save(netG.state_dict(), path_G)
                # path_P = f'./saved_model/netPrompt_{opt.dataset}_VADS.pth'
                # torch.save(netPrompt.state_dict(), path_P)
                # path_E = f'./saved_model/netE_{opt.dataset}_VADS.pth'
                # torch.save(netE.state_dict(), path_E)
                # path_ATT = f'./saved_model/netATT_{opt.dataset}_VADS.pth'
                # torch.save(netATT.state_dict(), path_ATT)
                

            print('GZSL: seen=%.3f, unseen=%.3f, h=%.3f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")
    # Zero-shot learning
    # Train ZSL classifier 
    
    if opt.final_classifier == 'softmax':
        zsl = classifier_zero.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                        data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, \
                        generalized=False, final_classifier='softmax', netVis=netE, netATT=None, dec_size=opt.attSize, dec_hidden_size=(opt.latensize*2),_att = att.cpu())
        acc = zsl.acc
        if best_zsl_acc <= acc:
            best_zsl_epoch= epoch
            best_zsl_acc = acc
        print('ZSL: unseen accuracy=%.4f' % (acc))
    if epoch % 10 == 0:
        print('GZSL: epoch=%d, best_seen=%.3f, best_unseen=%.3f, best_h=%.3f' % (best_gzsl_epoch, best_acc_seen, best_acc_unseen, best_gzsl_acc))
        print('ZSL: epoch=%d, best unseen accuracy=%.4f' % (best_zsl_epoch, best_zsl_acc))
    
    
    # reset G to training mode
    netG.train()  
    netPrompt.train() 
    wandb.log({
            'epoch': epoch,
            'loss_D': D_cost.item(),
            'loss_G':G_cost.item(),
            # 'Wasserstein_D': Wasserstein_D.item(),
            'acc_unseen': gzsl_cls.acc_unseen,
            'acc_seen': gzsl_cls.acc_seen,
            'H': gzsl_cls.H,
            'acc_zs': acc,
            'best_acc_unseen': best_acc_unseen,
            'best_acc_seen': best_acc_seen,
            'best_H': best_gzsl_acc,
            'best_acc_zs': best_zsl_acc
        })     
    
print('softmax: feature(X):2048')
print(time.strftime('ending time:%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)

