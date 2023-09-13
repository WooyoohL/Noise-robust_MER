
import torch
import torch.nn as nn
import os
import json
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler

from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from models.networks.shared import SharedEncoder
from models.utils import CMD


class UttShared002Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'CMD', 'Diff']
        self.modality = opt.modality
        self.model_names = ['C', 'Shared', 'A_spec', 'A_inv', 'L_spec', 'L_inv', 'V_spec', 'V_inv', 'Diff']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        self.netC = FcClassifier(2 * cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)

        self.netShared = SharedEncoder(opt)

        # 视频和音频是使用RNN，文本使用TextCNN
        # acoustic model
        if 'A' in self.modality:
            # self.model_names.append('A')
            self.netA_spec = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            self.netA_inv = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # lexical model
        if 'L' in self.modality:
            # self.model_names.append('L')
            self.netL_spec = TextCNN(opt.input_dim_l, opt.embd_size_l)
            self.netL_inv = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # visual model
        if 'V' in self.modality:
            # self.model_names.append('V')
            self.netV_spec = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            self.netV_inv = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)

        self.loss_cmd_func = CMD()

        # diffusion model
        self.netDiff = UNet2DConditionModel(cross_attention_dim=384,block_out_channels=(64, 128, 256, 256),
                                            addition_embed_type_num_heads=32, attention_head_dim=4)
        # block_out_channels=(64, 128, 256, 256),addition_embed_type_num_heads=64,attention_head_dim: Union=8
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        # self.netlin1 = nn.Linear(in_features=96, out_features=64)



        self.netDiff.cuda()
        # self.netlin1.cuda()

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if 'A' in self.modality:
            self.acoustic = input['A_feat'].float().to(self.device)
        if 'L' in self.modality:
            self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in self.modality:
            self.visual = input['V_feat'].float().to(self.device)
        
        self.label = input['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        final_embd = []
        final_shared = []
        # print(self.acoustic.size(),self.lexical.size(),self.visual.size())
        if 'A' in self.modality:
            self.feat_A = self.netA_inv(self.acoustic)  # netA：单层RNN
            self.feat_shared_A = self.netShared(self.feat_A)
            self.feat_A_spec = self.netA_spec(self.acoustic)
            final_embd.append(self.feat_A_spec)
            final_shared.append(self.feat_shared_A)

        if 'L' in self.modality:
            self.feat_L = self.netL_inv(self.lexical)   # netL：TextCNN
            self.feat_shared_L = self.netShared(self.feat_L)
            self.feat_L_spec = self.netL_spec(self.lexical)
            final_embd.append(self.feat_L_spec)
            final_shared.append(self.feat_shared_L)
        
        if 'V' in self.modality:
            self.feat_V = self.netV_inv(self.visual)
            self.feat_shared_V = self.netShared(self.feat_V)
            self.feat_V_spec = self.netV_spec(self.visual)    # netV：单层RNN
            final_embd.append(self.feat_V_spec)
            final_shared.append(self.feat_shared_V)
        
        # get model outputs
        self.feat = torch.cat(final_embd, dim=-1)   # 简单拼接
        self.feat_shared = torch.cat(final_shared, dim=-1)

        self.fusion_diff_input = self.feat.reshape(1, 4, 64, 96)   # batch 的位置可以不是1吗？
        self.encode_inv = self.feat_shared.reshape(1, 64, 384)

        self.noise = torch.randn_like(self.fusion_diff_input)
        self.bsz = self.fusion_diff_input.shape[0]
        self.timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (self.bsz,),
                                       device=self.fusion_diff_input.device)
        self.timesteps = self.timesteps.long()
        self.noisy_latents = self.noise_scheduler.add_noise(self.fusion_diff_input, self.noise, self.timesteps)
        self.diff_model_pred = self.netDiff(self.noisy_latents, self.timesteps,
                                            encoder_hidden_states=self.encode_inv).sample
        self.diff_model_pred = self.diff_model_pred + self.fusion_diff_input
        self.diff_model_pred2 = self.diff_model_pred.reshape(192, 128)
        self.logits, self.ef_fusion_feat = self.netC(torch.cat([self.feat, self.feat_shared], dim=-1)) # 两层128维的全连接层
        # self.logits, self.ef_fusion_feat = self.netC(self.feat) # 两层128维的全连接层
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.feat = self.feat.reshape(192,128)
        self.loss_Diff = F.mse_loss(self.feat.float(), self.diff_model_pred2.float())

        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_CMD = self.get_cmd_loss()
        loss = self.loss_CE + self.loss_CMD + 100 * self.loss_Diff
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 

    def get_cmd_loss(self, ):

        if not self.opt.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd_func(self.feat_shared_L, self.feat_shared_V, 5)
        loss += self.loss_cmd_func(self.feat_shared_L, self.feat_shared_A, 5)
        loss += self.loss_cmd_func(self.feat_shared_A, self.feat_shared_V, 5)
        loss = loss / 3.0

        return loss