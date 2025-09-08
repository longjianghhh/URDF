import torch
import torch.nn as nn
import torch.nn.functional as F
# from audtorch.metrics.functional import pearsonr

class Loss(nn.Module):
    def __init__(self, t, Nlabel,device):
        super(Loss, self).__init__()

        self.Nlabel = Nlabel
        self.t = t
        self.device = device
        self.CE = nn.CrossEntropyLoss(reduction="sum")
        self.mse = nn.MSELoss()
        self.criterion= nn.CrossEntropyLoss(reduction="sum")
        # self.lable_c=dep_graph#torch.mean(dep_graph,dim=1)
        
        # self.Positive=torch.exp(-dep_graph)
        # self.Negative=1-torch.exp(dep_graph-1)

    def label_graph2(self, emb, inc_labels, inc_L_ind):
        # label guide the embedding feature
        cls_num = inc_labels.shape[-1]
        valid_labels_sum = torch.matmul(inc_L_ind.float(), inc_L_ind.float().T) #[n, n] 

        # graph = torch.matmul(inc_labels, inc_labels.T).fill_diagonal_(0)
        graph = (torch.matmul(inc_labels, inc_labels.T).mul(valid_labels_sum) / (torch.matmul(inc_labels, inc_labels.T).mul(valid_labels_sum)+100)).fill_diagonal_(0)
        # print((graph>0.1).sum(),graph.shape)
        # assert torch.sum(torch.isnan(graph)).item() == 0
        graph = torch.clamp(graph,min=0,max=1.)
        emb = F.normalize(emb, p=2, dim=-1)
        # graph = graph.mul(graph>0.2)
        # graph = (inc_labels.mm(inc_labels.T))
        # graph = 0.5*(graph+graph.t())¸
        
        loss = 0
        Lap_graph  = torch.diag(graph.sum(1))- graph
        loss = torch.trace(emb.t().mm(Lap_graph).mm(emb))/emb.shape[0]
        return loss/emb.shape[0] #loss/number of views




    def wmse_loss(self,input1, target, weight, reduction='mean'):
        # ret = (torch.diag(weight).mm(target - input)) ** 2
        # ret = torch.mean(ret)
        
        ret1 = (torch.diag(weight).mm(target - input1)) ** 2
        ret1 = torch.mean(ret1)
      
        return ret1#(ret+ret1)/2 #ret,ret1#

    def cont_loss(self,S,V,inc_V_ind):
        loss_Cont = 0
        if isinstance(S,list):
            S = torch.stack(S,1) #[n v d]

        if isinstance(V,list):
            V = torch.stack(V,1) #[n v d]
        for i in range(S.size(0)):
            loss_Cont += self.forward_contrast(S[i], V[i], inc_V_ind[i,:])
        return loss_Cont
    
    def calculate_orthogonal_regularization_F(self,comm_feature,view_features_dict,inc_V_ind):
            loss = 0.0
   
            for i, view_feature in enumerate(view_features_dict):
                non_missing_indices = torch.nonzero(inc_V_ind[:,i]).squeeze()
                view_feature1 = F.normalize(view_feature[non_missing_indices]) 
                comm_feature1 = F.normalize(comm_feature[non_missing_indices]) 
                item = view_feature1.mm(comm_feature1.T)
                # item=(torch.diag(inc_V_ind[:,i])).mm(item).mm(torch.diag(inc_V_ind[:,i]))
                item = item ** 2
                item = item.mean()
                # item = item ** 0.5
                loss += item#/(len(non_missing_indices)*len(non_missing_indices))
            loss /= (len(view_features_dict))
            return loss
    
    def forward_contrast(self, si, vi, wei):
        ## S1 S2 [v d]
        si = si[wei.bool()]
        vi = vi[wei.bool()]
        n = si.size(0)
        N = 2 * n
        if n <= 1:
            return 0
        si = F.normalize(si, p=2, dim=1)
        vi = F.normalize(vi, p=2, dim=1)
        if si.shape[0]<=1 and vi.shape[0]<=1:
            return 0

        svi = torch.cat((si, vi), dim=0)

        sim = torch.matmul(svi, svi.T)
        # sim = (sim/self.t).exp()
        # print(sim)

        pos_mask = torch.zeros((N, N),device=sim.device)
        pos_mask[:n,:n] = torch.ones((n, n),device=sim.device)
        neg_mask = 1-pos_mask
        pos_mask = pos_mask.fill_diagonal_(0)
        neg_mask = neg_mask.fill_diagonal_(0)
        # neg_mask[n:N,n:N] = torch.zeros((n, n),device=sim.device)
        pos_pairs = sim.masked_select(pos_mask.bool())
        neg_pairs = sim.masked_select(neg_mask.bool())
        # prop = torch.exp(pos_pairs).mean()/(torch.exp(pos_pairs).mean()+torch.abs(torch.exp(neg_pairs)).mean())
        # loss = -torch.log(prop)
        loss = (neg_pairs).square().mean()/(((pos_pairs+1+1e-6)/2).mean())
        # loss = (neg_pairs).square().mean()/(pos_pairs).square().mean()
        # target = torch.eye(N,device=sim.device)
        # target[:n,:n] = torch.ones((n, n),device=sim.device)
        # loss = (-target.mul(torch.log((sim+1)/2+1e-6))-(1-target).mul(torch.log(1-sim.square()+1e-6))).mean()

        # assert torch.sum(torch.isnan(loss)).item() == 0
        return loss/2

   
    
 
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,epoch,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0

        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))      
        res1=torch.abs(sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5)))      
        res11=res1.mul(inc_L_ind)        
        res12=res1.mul(1-inc_L_ind)  
        
        meanloss = torch.sum(res11, dim=1, keepdim=True)/torch.sum(inc_L_ind, dim=1, keepdim=True)
        pso_inc_L_ind= (res12 < meanloss).float()
        pso_inc_L_ind=pso_inc_L_ind.mul(1-inc_L_ind)
        pso_inc_L_ind=pso_inc_L_ind+inc_L_ind
        res2=res1.mul(pso_inc_L_ind) 
        
        # non_mask=res3!=0
        # non_val=res3[non_mask]
        # #标准差
        # # mean_dev=torch.mean(non_val)
        # # std_dev=torch.std(non_val)
        # # threshold=mean_dev+2*std_dev
        # #IQR:
        # Q1=torch.quantile(non_val, 0.25)
        # Q3=torch.quantile(non_val, 0.75)
        # IQR=Q3-Q1
        # threshold=Q3+1.5*IQR
        
        # res0=torch.abs(sub_target.mul(torch.log(1-target_pre + 1e-5)) \
        #                                         + (1-sub_target).mul(torch.log(target_pre + 1e-5)))
        # res3[res3>=threshold]=res0[res3>=threshold]
        
        # res=res2+res3
        
        if reduction=='mean':
            # if epoch<=10:
            #     return torch.sum(res2)/torch.sum(inc_L_ind)
            # return torch.mean(res)
            if epoch<=30:
                return torch.sum(res11)/torch.sum(inc_L_ind)
            
            else:
                return torch.sum(res2)/torch.sum(pso_inc_L_ind)
            #     # print("哈哈哈")
            #     # A = (target_pre > self.Positive).float()
            #     # B = (target_pre < self.Negative).float()
            #     # psu=(A+B)*inc_L_ind
            #     # res1=torch.abs((A.mul(torch.log(target_pre + 1e-5)) \
            #     #                                         + (1-A).mul(torch.log(1 - target_pre + 1e-5))).mul(psu))      
                
            #     # res2=(self.lable_c*A+(1-self.lable_c)*(1-A)).mul(psu)     
            #     # # print(torch.sum(res2)/torch.sum(psu))
            #     # return torch.sum(res)/torch.sum(inc_L_ind)+torch.sum(res1)/torch.sum(psu)#-torch.sum(res2)/torch.sum(psu)
                
        
                
            
            
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res

    def forward_contrast1(self, v1_c,v1_s, v2_c,v2_s, we1, we2):
        mask_miss_inst = we1.mul(we2).bool() # mask the unavailable instances

        v1_c = v1_c[mask_miss_inst]
        v1_s = v1_s[mask_miss_inst]
    
        v2_c = v2_c[mask_miss_inst]
        v2_s = v2_s[mask_miss_inst]
        
        vc= torch.cat((v1_c, v2_c), dim=0)
        
        vs= torch.cat((v1_s, v2_s), dim=0)
        
        n = vc.size(0)
        N = vc.size(0)+vs.size(0)
        if n == 0:
            return 0
        vc = F.normalize(vc, p=2, dim=1) #normalize two vectors
        vs = F.normalize(vs, p=2, dim=1)
        
        mask = torch.ones((n, N)) # get mask
        mask = mask.fill_diagonal_(0)        
        mask = mask.bool()
        
        h = torch.cat((vc, vs), dim=0)
        sim_mat = torch.matmul(vc, h.T) / self.t
        positive_pairs=torch.diag(mask, 0).reshape(n, 1).to(self.device)
        negative_pairs = sim_mat[mask].reshape(n, -1)
        
        targets = torch.zeros(n).to(positive_pairs.device).long()
        logits = torch.cat((positive_pairs, negative_pairs), dim=1).to(positive_pairs.device)
        loss = self.criterion(logits, targets)
        return loss/n

    
