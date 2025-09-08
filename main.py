import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model import get_model
import evaluation
import torch
import numpy as np
from myloss import Loss
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import copy
import time

def loss_view_classifier(sub_target,y_specific,fan_sub_target,sub_obrT,We,epoch):
    bool_we=We.bool()
    
    loss_S=0
    if epoch>30:
        for i in range(6):
            res1=torch.abs(sub_target.mul(torch.log(y_specific[i] + 1e-5)) \
                                                    + (1-sub_target).mul(torch.log(1 - y_specific[i] + 1e-5)))      
            res11=res1.mul(sub_obrT)        
            res12=res1.mul(1-sub_obrT)  
            
            meanloss = torch.sum(res11, dim=1, keepdim=True)/torch.sum(sub_obrT, dim=1, keepdim=True)
            pso_inc_L_ind= (res12 < meanloss).float()
            pso_inc_L_ind=pso_inc_L_ind.mul(1-sub_obrT) +sub_obrT
            # pso_inc_L_ind=pso_inc_L_ind+sub_obrT
            res2=res1.mul(pso_inc_L_ind) 
           # print(torch.sum(pso_inc_L_ind[bool_we[:,i]]))
            loss_S=loss_S+torch.sum(res2[bool_we[:,i]])/torch.sum(pso_inc_L_ind[bool_we[:,i]])
        return  loss_S
    
    
    loss_SV0 = torch.mean(torch.abs(((sub_target.mul(torch.log(y_specific[0] + 1e-10)) \
                                    + fan_sub_target.mul(torch.log(1 - y_specific[0] + 1e-10))).mul(sub_obrT))[bool_we[:,0]]))
    loss_SV1 = torch.mean(torch.abs(((sub_target.mul(torch.log(y_specific[1] + 1e-10)) \
                                    + fan_sub_target.mul(torch.log(1 - y_specific[1] + 1e-10))).mul(sub_obrT))[bool_we[:,1]]))
    loss_SV2 = torch.mean(torch.abs(((sub_target.mul(torch.log(y_specific[2] + 1e-10)) \
                                    + fan_sub_target.mul(torch.log(1 - y_specific[2] + 1e-10))).mul(sub_obrT))[bool_we[:,2]]))
    loss_SV3 = torch.mean(torch.abs(((sub_target.mul(torch.log(y_specific[3] + 1e-10)) \
                                    + fan_sub_target.mul(torch.log(1 - y_specific[3] + 1e-10))).mul(sub_obrT))[bool_we[:,3]]))
    loss_SV4 = torch.mean(torch.abs(((sub_target.mul(torch.log(y_specific[4] + 1e-10)) \
                                    + fan_sub_target.mul(torch.log(1 - y_specific[4] + 1e-10))).mul(sub_obrT))[bool_we[:,4]]))
    loss_SV5 = torch.mean(torch.abs(((sub_target.mul(torch.log(y_specific[5] + 1e-10)) \
                                    + fan_sub_target.mul(torch.log(1 - y_specific[5] + 1e-10))).mul(sub_obrT))[bool_we[:,5]]))   
    loss_SV=loss_SV0+loss_SV1+ loss_SV2+loss_SV3+loss_SV4+ loss_SV5  
    
    # y_p0 = (y_specific[0] > 0.5).type(sub_target.dtype) 
    return loss_SV

def train(loader, model, loss_model, opt, sche, epoch,logger,sim_epochs):
    # assert torch.sum(torch.isnan(dep_graph)).item() == 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        label = label.to('cuda:0')
        # inc_L _ind = torch.ones_like(inc_L_ind)
        # print(sum(label.sum(axis=0)==0),label[:,label.sum(axis=0)==0].shape)
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        x_bar_list_c,target_pre, fusion_z, share_zs,y_s= model(data,inc_V_ind,mode='train',sigma=args.sigma)
        
       
        # loss_Cont = loss_model.cont_loss(share_zs,viewsp_zs,inc_V_ind)
        
        loss_CL = loss_model.weighted_BCE_loss(target_pre,label,inc_L_ind,epoch)
        loss_Cs=  loss_view_classifier(label,y_s,1-label,inc_L_ind,inc_V_ind,epoch) 
        
        loss_CL=loss_CL+loss_Cs#+loss_Cv
       
        loss_AE = 0
       
       
        for iv, x_bar_c in enumerate(x_bar_list_c):         
            loss_AE += loss_model.wmse_loss(x_bar_c, data[iv], inc_V_ind[:, iv])
                   
        loss = loss_CL + args.gamma * loss_AE
        
        
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.sum:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, 
                        losses=losses))
    # print("all0",all0)
    return losses,model

def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        x_bar_list_c, pred, fusion_z, _,_= model(data,inc_V_ind,mode='test',sigma=0)          
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        
        batch_time.update(time.time()- end)
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.sum:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results

def seed_torch(seed=1029):
# 	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args,file_path):
    seed_torch(42)
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = False,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
        cur_inc_L_ind= torch.tensor(train_dataset.cur_inc_L_ind).float().to('cuda:0')
          
        
        model = get_model(n_stacks=4,n_input=d_list,n_z=args.n_z,Nlabel=classes_num,device=device)
        # print(model)
        loss_model = Loss(0.2, classes_num,  device)#class_d,

        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)      
        scheduler = None
        

        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        
        sim_epochs = []
        for epoch in range(args.epochs):
            tt=time.time()
            train_losses,model = train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,logger,sim_epochs)

            val_results = test(val_dataloder,model,loss_model,epoch,logger)

            
            if val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.5>=static_res:   #adjust weight of each metric
                static_res = val_results[0]*0.5+val_results[2]*0.25+val_results[3]*0.5
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch
            train_losses_last = train_losses
            total_losses.update(train_losses.sum)
        model.load_state_dict(best_model_dict['model'])
        print("epoch",best_model_dict['epoch'])
        test_results = test(test_dataloder,model,loss_model,epoch,logger)
        if len(sim_epochs)>0:
            np.save(f'diction/{args.dataset}_feature.npy',torch.stack(sim_epochs,dim=0).numpy())
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme one_error coverage macAUC macro_f1 micro_f1 lr alpha beta gamma sigma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,3))+'+'+str(round(res.std,3)) for res in folds_results]
    res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma),str(args.sigma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k']) #here to select which dataset you want
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int) # here to set the repeat number  
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='10_final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e0) # not work here
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100) # here to set the repeat number  
    
    # Training args
    parser.add_argument('--n_z', type=int, default=512) # here to set the dimension
    parser.add_argument('--batch_size', type=int, default=128) # here to set the batch_size
    parser.add_argument('--alpha', type=float, default=1e-1) # not work here, set it below
    parser.add_argument('--beta', type=float, default=1e-1) # not work here, set it below
    parser.add_argument('--gamma', type=float, default=1e-1) # not work here, set it below
    parser.add_argument('--sigma', type=float, default=0.) # not work here, set it below

    
    args = parser.parse_args()
    if args.records_dir:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e0]#,0.1]
    alpha_list = [1e0]#,1e-1,1e-2,1e-3,1e-4,1e-5] # set 1e-4 for all datasets
    beta_list = [1e-4]#,1e-2,1e-3,1e-1,0,1]  # 1e-4 for others  1e-3 for mir
    gamma_list = [1e-1]#,1e-2,1e-3,1e-4,1e-5,1]
    sigma_list = [0.25]#0.25]
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                            args.mask_view_ratio) + '_LM_' +
                                            str(args.mask_label_ratio) + '_T_' + 
                                            str(args.training_sample_ratio) + '.txt')
                            args.file_path = file_path
                            # existed_params = filterparam(file_path,[-4,-3,-2,-1])
                            # if [args.alpha,args.beta,args.gamma,args.sigma] in existed_params:
                            #     print('existed param! alpha:{} beta:{} gamma:{} sigma:{}'.format(args.alpha,args.beta,args.gamma,args.sigma))
                            #     continue
                            main(args,file_path)