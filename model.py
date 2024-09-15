#author: akshitac8
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Encoder
class Encoder(nn.Module):

    def __init__(self, opt):

        super(Encoder,self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.latent_size
        layer_sizes[0] += latent_size
        self.fc1=nn.Linear(layer_sizes[0], layer_sizes[-1])
        self.fc3=nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None: x = torch.cat((x, c), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

#Decoder/Generator
# class Generator(nn.Module):

#     def __init__(self, opt):

#         super(Generator,self).__init__()

#         layer_sizes = opt.decoder_layer_sizes
#         latent_size=opt.latent_size
#         input_size = latent_size * 2
#         self.fc1 = nn.Linear(input_size, layer_sizes[0])
#         self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         self.sigmoid=nn.Sigmoid()
#         self.apply(weights_init)

#     def _forward(self, z, c=None):
#         z = torch.cat((z, c), dim=-1)
#         x1 = self.lrelu(self.fc1(z))
#         x = self.sigmoid(self.fc3(x1))
#         self.out = x1
#         return x

#     def forward(self, z, a1=None, c=None, feedback_layers=None):
#         if feedback_layers is None:
#             return self._forward(z,c)
#         else:
#             z = torch.cat((z, c), dim=-1)
#             x1 = self.lrelu(self.fc1(z))
#             feedback_out = x1 + a1*feedback_layers
#             x = self.sigmoid(self.fc3(feedback_out))
#             return x

class Generator(nn.Module):

    def __init__(self, opt):

        super(Generator,self).__init__()

        layer_sizes = opt.decoder_layer_sizes
        latent_size=opt.latent_size
        #input_size = latent_size * 2 #+ opt.dim_bias
        input_size = latent_size + opt.dim_bias
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc3 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid=nn.Sigmoid()
        self.apply(weights_init)

    def _forward(self, z, c=None):
        z = torch.cat((z, c), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc3(x1))
        self.out = x1
        return x

    def forward(self, z, a1=None, c=None, feedback_layers=None):
        if feedback_layers is None:
            return self._forward(z,c)
        else:
            z = torch.cat((z, c), dim=-1)
            x1 = self.lrelu(self.fc1(z))
            feedback_out = x1 + a1*feedback_layers
            x = self.sigmoid(self.fc3(feedback_out))
            return x


#conditional discriminator for inductive
class Discriminator(nn.Module):
    def __init__(self, opt): 
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h
        
#Feedback Modules
class Feedback(nn.Module):
    def __init__(self,opt):
        super(Feedback, self).__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)
    def forward(self,x):
        self.x1 = self.lrelu(self.fc1(x))
        h = self.lrelu(self.fc2(self.x1))
        return h

# class AtttoAtt(nn.Module):
#     def __init__(self,opt):
#         super(AtttoAtt, self).__init__()
#         self.fc1 = nn.Linear(312, 1024)
#         self.fc2 = nn.Linear(1024, 312)
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         self.apply(weights_init)
#     def forward(self,x):
#         self.x1 = self.lrelu(self.fc1(x))
#         h = self.lrelu(self.fc2(self.x1))
#         return h

class AtttoAtt(nn.Module):
    def __init__(self,opt):
        super(AtttoAtt,self).__init__()
        self.linear1 = nn.Linear(opt.dim, opt.dim * 4)
        self.linear2 = nn.Linear(opt.dim * 4, opt.dim)
        
        self.fc1 = nn.Linear(opt.dim, opt.dim//4) 
        self.fc2 = nn.Linear(opt.dim//4, opt.dim)    
        
        self.activation = nn.ReLU()
        self.apply(weights_init)

    def forward(self,x):
        w = F.sigmoid(self.fc2(self.activation(self.fc1(x))))
        x = self.linear2(self.activation(self.linear1(x)))
        x = x * w
        return x

class FR(nn.Module):
    def __init__(self, opt, attSize):
        super(FR, self).__init__()
        self.embedSz = 0
        self.hidden = None
        self.lantent = None
        self.latensize=opt.latensize
        self.attSize = opt.attSize
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize*2)
        # self.encoder_linear = nn.Linear(opt.resSize, opt.latensize*2)
        self.discriminator = nn.Linear(opt.attSize, 1)
        self.classifier = nn.Linear(opt.attSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, feat, train_G=False,att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        self.lantent = self.fc3(self.hidden)
        mus,stds = self.lantent[:,:self.attSize],self.lantent[:,self.attSize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        h= encoder_out
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))
        if self.sigmoid is not None:
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        return mus, stds, dis_out, pred, encoder_out, h
    
    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu







class VistoAtt(nn.Module):
    def __init__(self, opt):
        super(VistoAtt, self).__init__()
        self.linear1 = nn.Linear(opt.visdim, opt.visdim * 2)
        self.linear2 = nn.Linear(opt.visdim * 2, opt.visdim * 2)
        self.linear3 = nn.Linear(opt.visdim * 2, opt.visdim) 
        
        self.fc1 = nn.Linear(opt.visdim, opt.visdim//2) 
        self.fc2 = nn.Linear(opt.visdim//2, opt.visdim//2)
        self.fc3 = nn.Linear(opt.visdim//2, opt.dim)    
        
        self.activation = nn.ReLU()
        self.sigmoid = None#nn.Sigmoid()
        self.apply(weights_init) 

    def forward(self,x):
        out  = F.sigmoid(self.linear3(self.activation(self.linear2(self.activation(self.linear1(x))))))
        x = out + x
        x = self.fc3(self.activation(self.fc2(self.activation(self.fc1(x))))) 
        if self.sigmoid is not None:
            h = self.sigmoid(x)
        else:
            h = x/x.pow(2).sum(1).sqrt().unsqueeze(1).expand(x.size(0),x.size(1)) 

        return h



class Map_Net(nn.Module):
    def __init__(self, opt):
        super(Map_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, x):
        latent_map = self.relu(self.fc1(x))
        out_z = F.normalize(self.fc2(latent_map), dim=1)
        return latent_map, out_z

class Map_VisNet(nn.Module):
    def __init__(self, opt):
        super(Map_VisNet, self).__init__()

        self.fc1 = nn.Linear(opt.embedSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, x):
        vis_map = self.relu(self.fc1(x))
        vis_out = F.normalize(self.fc2(vis_map),dim=1)
        return vis_out

class Map_Vis_att_update(nn.Module):
    def __init__(self, opt):
        super(Map_Vis_att_update, self).__init__()

        self.fc1 = nn.Linear(opt.embedSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, 312)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)
        self.att_update = nn.Sequential(
            nn.Linear(312,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,312)
            )
        for name,param in self.att_update.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param,mean=0,std=0.0)
        
        self.log_softmax_fun = nn.LogSoftmax(dim=-1)

    def forward(self, x,att):
        vis_map = self.relu(self.fc1(x))
        vis_out = F.normalize(self.fc2(vis_map),dim=1)
        self.update = F.normalize(att+self.att_update(att))
        s_pp = self.log_softmax_fun(torch.einsum('ki,bi->bk',self.update,vis_out))
        #s_pp = self.log_softmax_fun(torch.einsum('ki,bi->bk',att,vis_out))
        return vis_out,self.update,s_pp
        #return vis_out,None,s_pp
    
    def update_att(self):
        return self.update

class Att_update(nn.Module):
    def __init__(self, opt):
        super(Att_update, self).__init__()
        self.att_update = nn.Sequential(
            nn.Linear(opt.attSize,opt.attSize),
            nn.LeakyReLU()
        
            )
        
        

    def forward(self,att):
      
        self.update = F.normalize(att+self.att_update(att))
        return self.update
    
    def update_att(self):
        return self.update    


# class VistoAtt1(nn.Module):
#     def __init__(self, opt):
#         super(VistoAtt, self).__init__()
#         self.linear1 = nn.Linear(opt.visdim, opt.visdim * 2)
#         self.linear2 = nn.Linear(opt.visdim * 2, opt.visdim * 2)
#         self.linear3 = nn.Linear(opt.visdim * 2, opt.visdim) 
        
#         self.fc1 = nn.Linear(opt.visdim, opt.visdim//2) 
#         self.fc2 = nn.Linear(opt.visdim//2, opt.visdim//2)
#         self.fc3 = nn.Linear(opt.visdim//2, opt.dim)    
        
#         self.activation = nn.ReLU()
#         self.sigmoid = nn.Sigmoid() 

#     def forward(self,x):
#         out  = F.sigmoid(self.linear3(self.activation(self.linear2(self.activation(self.linear1(x))))))
#         x = out + x
#         x = self.fc3(self.activation(self.fc2(self.activation(self.fc1(x))))) 
#         if self.sigmoid is not None:
#             h = self.sigmoid(x)
#         else:
#             h = x/x.pow(2).sum(1).sqrt().unsqueeze(1).expand(x.size(0),x.size(1)) 

#         return h


class AttDec(nn.Module):
    def __init__(self, opt, attSize):
        super(AttDec, self).__init__()
        self.embedSz = 0
        self.fc1 = nn.Linear(opt.resSize + self.embedSz, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.hidden = None
        self.sigmoid = None
        self.apply(weights_init)

    def forward(self, feat, att=None):
        h = feat
        if self.embedSz > 0:
            assert att is not None, 'Conditional Decoder requires attribute input'
            h = torch.cat((feat,att),1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc3(self.hidden)
        if self.sigmoid is not None: 
            h = self.sigmoid(h)
        else:
            h = h/h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0),h.size(1))
        self.out = h
        return h

    def getLayersOutDet(self):
        #used at synthesis time and feature transformation
        return self.hidden.detach()
    



class Encoder_Prompt(nn.Module):
    def __init__(self,opt) :
        super(Encoder_Prompt,self).__init__()
        self.e = nn.Sequential(
            nn.Linear(768,2048),
            nn.ReLU(),
        )

        self.contrastive = nn.Linear(2048,512)

        self.mean = nn.Linear(2048,opt.dim_bias)
        self.log_var = nn.Linear(2048,opt.dim_bias)
    
    def forward(self,x):
        #x = torch.cat([x,z],dim=1)
        x = self.e(x)
        e = x
        contra = F.normalize(self.contrastive(x),dim=1)
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean,log_var,contra,e

class PromptsLearner(nn.Module):
    def __init__(self, opt):
        super(PromptsLearner,self).__init__()
        self.n_cls = opt.nclass_seen
        self.n_ctx = 1
        self.global_ctx = opt.global_ctx
        ctx_dim = opt.dim_bias
        self.opt = opt
        if self.global_ctx:
            ctx_vectors = torch.empty(self.n_ctx,ctx_dim).cuda()
        else:
            ctx_vectors = torch.empty(self.n_cls,ctx_dim).cuda()

        nn.init.normal_(ctx_vectors)

        self.ctx = nn.Parameter(ctx_vectors)

        self.net = nn.Sequential(
            nn.Linear(opt.dim_bias,4096),
            nn.ReLU(),
            nn.Linear(4096,opt.dim_bias)
        )
        # self.sigmoid = nn.Sigmoid()
        # self.cls = nn.Linear(opt.dim_bias, 150)
        # self.logic = nn.LogSoftmax(dim=1)

        self.mean = nn.Linear(opt.dim_bias,64)
        self.log_var = nn.Linear(opt.dim_bias,64)

    def forward(self,z=None,target=None,unseen_ctx=None):
        if self.global_ctx:
            if z is not None and unseen_ctx is None:
                ctx = self.ctx
                bias = self.net(z)
                bias = bias.unsqueeze(1)
               
                ctx = ctx.unsqueeze(0)
            
                ctx_shifted = ctx+bias
                prompt  = ctx_shifted.squeeze(1)

                # #prompt = ctx.expand(bias.shape[0],self.opt.dim_bias)
                #prompt = bias
                



            
            else:
                ctx = self.ctx
                prompt = ctx
        
        else:
            if z is not None and unseen_ctx is None:
                ctx = self.ctx
                bias = self.net(z)
                ctx = ctx[target]
                ctx_shifted = ctx+bias
                prompt  = ctx_shifted
            
            else:
                bias = self.net(z)
                ctx = unseen_ctx
                ctx_shifted = ctx#+bias
                prompt  = ctx_shifted
        
        #prompt_cls = self.logic(self.cls(prompt))
        mean = self.mean(prompt)
        log_var = self.log_var(prompt)
        return prompt,mean,log_var
        #return prompt,prompt_cls
    
    def get_seen_prompt(self):
        return self.ctx


# class BiasLearner(nn.Module):
#     def __init__(self, opt) :
#         super(BiasLearner,self).__init__()
#         self.net = nn.Sequential(nn.)

