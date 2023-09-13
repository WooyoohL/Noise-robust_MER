import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
from models.networks.shared import SharedEncoder
from Domiss import NoiseScheduler
from models.vae_model import ConditionVAE, vae_loss
from models.utt_shared_002_model import UttShared002Model


class NMERModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--pretrained_invariant_path', type=str,
                            help='where to load pretrained invariant encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--invariant_weight', type=float, default=1.0, help='weight of invariant loss')
        parser.add_argument('--share_weight', action='store_true',
                            help='share weight of forward and backward autoencoders')
        parser.add_argument('--image_dir', type=str, default='./invariant_image', help='models image are saved here')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['CE', 'invariant', 'vae']
        self.model_names = ['C', 'invariant',
                            'A_inv', 'A_spec', 'L_inv', 'L_spec', 'V_inv', 'V_spec',
                            'VAE']
        self.batch_size = opt.batch_size
        self.num_time_step = opt.num_time_step
        self.noise_type = opt.noise_type

        # noise_scheduler
        self.noise_scheduler = NoiseScheduler(noise_type=self.noise_type, num_time_steps=self.num_time_step)

        # acoustic model
        self.netA_inv = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netA_spec = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)

        # lexical model
        self.netL_inv = TextCNN(opt.input_dim_l, opt.embd_size_l)
        self.netL_spec = TextCNN(opt.input_dim_l, opt.embd_size_l)

        # visual model
        self.netV_inv = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.netV_spec = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)

        # 分类层
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        self.netC = FcClassifier(384, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                 use_bn=True)

        self.netinvariant = SharedEncoder(opt)
        self.netVAE = ConditionVAE()


        if self.isTrain:
            self.load_pretrained_encoder(opt)
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.invariant_weight = opt.invariant_weight
            self.cycle_weight = opt.cycle_weight
        else:
            self.load_pretrained_encoder(opt)

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        image_save_dir = os.path.join(opt.image_dir, opt.name)
        image_save_dir = os.path.join(image_save_dir, str(opt.cvNo))
        self.predict_image_save_dir = os.path.join(image_save_dir, 'predict')
        self.invariant_image_save_dir = os.path.join(image_save_dir, 'invariant')
        self.loss_image_save_dir = os.path.join(image_save_dir, 'loss')
        if not os.path.exists(self.predict_image_save_dir):
            os.makedirs(self.predict_image_save_dir)
        if not os.path.exists(self.invariant_image_save_dir):
            os.makedirs(self.invariant_image_save_dir)
        if not os.path.exists(self.loss_image_save_dir):
            os.makedirs(self.loss_image_save_dir)

    # 加载预训练Encoder，
    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False  # teacher model should be in test mode
        pretrained_config.gpu_ids = opt.gpu_ids  # set gpu to the same
        self.pretrained_encoder = UttShared002Model(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()

    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netA_spec.load_state_dict(f(self.pretrained_encoder.netA_spec.state_dict()))
            self.netV_spec.load_state_dict(f(self.pretrained_encoder.netV_spec.state_dict()))
            self.netL_spec.load_state_dict(f(self.pretrained_encoder.netL_spec.state_dict()))
            self.netA_inv.load_state_dict(f(self.pretrained_encoder.netA_inv.state_dict()))
            self.netV_inv.load_state_dict(f(self.pretrained_encoder.netV_inv.state_dict()))
            self.netL_inv.load_state_dict(f(self.pretrained_encoder.netL_inv.state_dict()))
            self.netinvariant.load_state_dict(f(self.pretrained_encoder.netShared.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.acoustic = acoustic = input['A_feat'].float().to(self.device)
        self.lexical = lexical = input['L_feat'].float().to(self.device)
        self.visual = visual = input['V_feat'].float().to(self.device)
        self.missing_index = input['missing_index'].long().to(self.device)  # [a,v,l]

        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)

        self.A_miss, _ = self.noise_scheduler.add_noise(self.acoustic, self.A_miss_index)
        self.V_miss, _ = self.noise_scheduler.add_noise(self.visual, self.V_miss_index)
        self.L_miss, _ = self.noise_scheduler.add_noise(self.lexical, self.L_miss_index)

        if self.isTrain:
            self.label = input['label'].to(self.device)
        else:
            pass


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss = self.netA_inv(self.A_miss)  # 缺失视频特征
        self.feat_A_miss_spec = self.netA_spec(self.A_miss)

        self.feat_L_miss = self.netL_inv(self.L_miss)
        self.feat_L_miss_spec = self.netL_spec(self.L_miss)

        self.feat_V_miss = self.netV_inv(self.V_miss)
        self.feat_V_miss_spec = self.netV_spec(self.V_miss)

        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss_spec, self.feat_L_miss_spec, self.feat_V_miss_spec], dim=-1)
        self.feat_A_invariant = self.netinvariant(self.feat_A_miss)
        self.feat_L_invariant = self.netinvariant(self.feat_L_miss)
        self.feat_V_invariant = self.netinvariant(self.feat_V_miss)
        self.invariant_miss = torch.cat([self.feat_A_invariant, self.feat_L_invariant, self.feat_V_invariant], dim=-1)


        self.x_recon, self.mu, self.logvar = self.netVAE(self.feat_fusion_miss, self.invariant_miss)
        # self.x_recon = self.x_recon + self.feat_fusion_miss
        self.logits, _ = self.netC(self.x_recon)

        self.pred = F.softmax(self.logits, dim=-1)
        # for training
        if self.isTrain:
            with torch.no_grad():
                self.T_embd_A = self.pretrained_encoder.netA_spec(self.acoustic)
                self.T_embd_L = self.pretrained_encoder.netL_spec(self.lexical)
                self.T_embd_V = self.pretrained_encoder.netV_spec(self.visual)
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)

                embd_A = self.pretrained_encoder.netA_inv(self.acoustic)
                embd_L = self.pretrained_encoder.netL_inv(self.lexical)
                embd_V = self.pretrained_encoder.netV_inv(self.visual)

                embd_A_invariant = self.pretrained_encoder.netShared(embd_A)
                embd_L_invariant = self.pretrained_encoder.netShared(embd_L)
                embd_V_invariant = self.pretrained_encoder.netShared(embd_V)
                self.invariant = torch.cat([embd_A_invariant, embd_L_invariant, embd_V_invariant], dim=-1)


    def backward(self):
        """Calculate the loss for back propagation"""
        # 分类损失
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)

        self.loss_vae = vae_loss(self.T_embds, self.x_recon, self.mu, self.logvar)
        # 占位，共性特征损失
        self.loss_invariant = self.invariant_weight * self.criterion_mse(self.invariant, self.invariant_miss)

        loss = self.loss_CE + self.loss_invariant + self.loss_vae
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
