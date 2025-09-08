import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from random import sample
from einops import rearrange
from transformer import *

class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        # print(enc_h3.shape)
        z = self.z_b0(self.z_layer(enc_h3))
        return z


class decoder1(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder1, self).__init__()
        self.dec_0 = Linear(2*n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
class decoder2(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder2, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
def information_fusion(y_view,We):
    bool_we=We.bool()
    
    mean_vectors = []
    for i in range(len(y_view)):
        # y_v=-y_view[i] *torch.log(y_view[i])
        y_v=1-y_view[i] * (1-y_view[i])      
        mean_vector = torch.mean(y_v, dim=1, keepdim=True)  
        mean_vectors.append(mean_vector)   
    
    concatenated_tensor = torch.cat(mean_vectors, dim=1)
    concatenated_tensor=concatenated_tensor*We
    
    row_sums = torch.sum(concatenated_tensor, dim=1, keepdim=True)
    weight_m= concatenated_tensor/ row_sums
    
    
    return weight_m


class MLP(nn.Module):
    def __init__(self,input_dim,hid_dimen1,hid_dimen3):
        super(MLP,self).__init__()
        self.Mlp =  nn.Sequential(
        nn.Linear(input_dim,hid_dimen1),
        nn.Tanh(),       
        nn.Linear(hid_dimen1, hid_dimen3),
        nn.Tanh(),
        )

    def forward(self, x):
        mlp = self.Mlp(x)
        return mlp

class net(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(net, self).__init__()


        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.ModuleList([decoder1(n_input[i], dims[i], 1*n_z) for i in range(len(n_input))])
        self.encoder2_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder2_list = nn.ModuleList([decoder2(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.regression = Linear(1*n_z, nLabel)
        self.regression_s = Linear(1*n_z, nLabel)
        self.regression_v = Linear(1*n_z, nLabel)
        self.act = nn.Sigmoid()
        self.nLabel = nLabel
        self.BN = nn.BatchNorm1d(n_z)
        self.nz=n_z
        self.num_views=len(n_input)
        
        self.classifier_view=Linear(n_z, self.num_views)#MLP(self.nz,512,self.num_views)
  
       
    def forward(self, mul_X, we,mode,sigma):
        # dep_graph = torch.eye(self.nLabel,device=we.device).float()
        batch_size = mul_X[0].shape[0]
        summ = 0
        prop = sigma
        share_zs = []
        if mode =='train':
            for i,X in enumerate(mul_X):
                mask_len = int(prop*X.size(-1))            
                    
                #点失活
                mask = torch.ones_like(X)
                for j in range(mask.shape[0]):
                    zero_indices = torch.randperm(mask.shape[1])[:mask_len]
                    mask[j, zero_indices] = 0
                mul_X[i] = mul_X[i].mul(mask)

        for enc_i, enc in enumerate(self.encoder_list):        
            missing_indices = torch.nonzero(we[:,enc_i]== 0).squeeze() #先编码后填充
            z_i = enc(mul_X[enc_i])                        
            # z_i[missing_indices]=0
            share_zs.append(z_i)
  
        y_s=[]
        for i in range(len(mul_X)):
            y1=self.act(self.regression_s(F.relu(share_zs[i])))
            y_s.append(y1)
        weight_c=information_fusion(y_s,we)
        s_z=torch.zeros_like(share_zs[0])
        for i in range(len(share_zs)):
            
            first_column_of_W = weight_c[:, i]     
            result = share_zs[i] * first_column_of_W.view(-1, 1)
            s_z=s_z+result
        
        
        z = s_z.mul(s_z.sigmoid_()) 
        # z = self.BN(z)
        z = F.relu(z)
        
        
        x_bar_list_c = []
        for dec_i, dec in enumerate(self.decoder2_list):
            # x_bar_list.append(dec(share_zs[dec_i]+viewsp_zs[dec_i]))
            x_bar_list_c.append(dec(z))
              
        logi = self.regression(z) #[n c]     
        yLable = self.act(logi)
           
        return x_bar_list_c,yLable, z, share_zs,y_s
def get_model(n_stacks,n_input,n_z,Nlabel,device):
    model = net(n_stacks=n_stacks,n_input=n_input,n_z=n_z,nLabel=Nlabel).to(device)
    return model