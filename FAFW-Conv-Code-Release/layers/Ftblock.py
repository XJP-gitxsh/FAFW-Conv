#import math
#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from mamba_ssm import Mamba
#from einops import rearrange
#from mpmath.identification import transforms
#from timm.models.layers import trunc_normal_
#
#
#from layers.RevIN import RevIN
#
#class Flatten_Head(nn.Module):
#    def __init__(self, n_vars, nf, target_window, head_dropout=0):
#        super().__init__()
#
#
#        self.n_vars = n_vars
#
#
#
#        self.flatten = nn.Flatten(start_dim=-2)
#        self.linear = nn.Linear(nf, target_window)
#        self.dropout = nn.Dropout(head_dropout)
#
#    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
#        x = self.flatten(x)
#        x = self.linear(x)
#        x = self.dropout(x)
#        #x:[bs,nvars,target_window]
#        return x
#
#class MLPRegression(nn.Module):
#    def __init__(self, input_size, hidden_size, output_size):
#        super(MLPRegression, self).__init__()
#        self.layer1 = nn.Linear(input_size, hidden_size)
#        self.relu1 = nn.ReLU()
#        self.layer2 = nn.Linear(hidden_size, hidden_size)
#        self.relu2 = nn.ReLU()
#        self.layer3 = nn.Linear(hidden_size, output_size)
#
#
#    def forward(self, x):
#        x = self.relu1(self.layer1(x))
#        x = self.relu2(self.layer2(x))
#        x = self.layer3(x)
#        return x
#
#
#class Encoder(nn.Module):
#    def __init__(self,d_model,d_state,d_conv,expand,dropout):
#        super(Encoder, self).__init__()
#
#        self.d_model = d_model
#        self.d_state = d_state
#        self.d_conv = d_conv
#        self.expand =expand
#
#        self.mamba = Mamba(d_model=self.d_model,d_state=self.d_state,d_conv=self.d_conv, expand=self.expand)
#
#
#    def forward(self, x):
#
#        x = self.mamba(x)
#
#
#        return x
#
#
#class Mamba_Backbone(nn.Module):
#    def __init__(self,patch_num,patch_len,enc_in,d_model,d_state,d_conv,expand,dropout, learn_pe=True):
#        super(Mamba_Backbone, self).__init__()
#        self.patch_num = patch_num
#        self.patch_len = patch_len
#        self.d_model = d_model
#        self.d_state = d_state
#        self.dropout = dropout
#        self.d_conv = d_conv
#        self.expand = expand
#
#
#        self.learnable_matrix = nn.Parameter(torch.randn(patch_num, d_model))
#
#        self.W_P = nn.Linear(self.patch_len, self.d_model)
#        self.mamba = Mamba(d_model=self.d_model,d_state=self.d_state,d_conv=self.d_conv, expand=self.expand)
#
#        
#
#
#
#
#    def forward(self, x,xmark_ec):#x:[B,E,patch_num,patch_len]
#        _,E,_,_ = x.shape
#
#        x = self.W_P(x)#x：[B,E,patch_num,d_model]
#
#
#
#        x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3))#x:[B*E,patch_num,d_model]
#        x = self.mamba(x)
#
#        #x = self.encoder(x+self.W_pos)#x:[B*E,patch_num,d_model]
#        x = torch.reshape(x, (-1, E, x.shape[-2], x.shape[-1]))#x:[B*E,patch_num,d_model]
#        x = x.permute(0,1,3,2)#x:[B*E,d_model,patch_num]
#
#
#        return x
#
#
#class DataEmbedding_inverted(nn.Module):
#    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
#        super(DataEmbedding_inverted, self).__init__()
#        self.value_embedding = nn.Linear(c_in, d_model)
#        self.dropout = nn.Dropout(p=dropout)
#
#    def forward(self, x, x_mark):
#        x = x.permute(0, 2, 1)
#        # x: [Batch Variate Time]
#        if x_mark is None:
#            x = self.value_embedding(x)
#        else:
#            # the potential to take covariates (e.g. timestamps) as tokens
#            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
#        # x: [Batch Variate d_model]
#        return self.dropout(x)
#
#class moving_avg(nn.Module):
#    """
#    Moving average block to highlight the trend of time series
#    """
#    def __init__(self, kernel_size, stride):
#        super(moving_avg, self).__init__()
#        self.kernel_size = kernel_size#移动平均块的参数kernel_size，stride
#        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)#平均池化层，计算移动平均值
#
#    def forward(self, x):
#        # padding on the both ends of time series
#        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
#        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
#        x = torch.cat([front, x, end], dim=1)
#        x = self.avg(x.permute(0, 2, 1))#将填充后的时间序列通过 permute 方法进行维度变换，以适应平均池化层的输入格式要求
#        x = x.permute(0, 2, 1)
#        return x
#
#class series_decomp(nn.Module):
#    """
#    Series decomposition block
#    """
#    def __init__(self, kernel_size):
#        super(series_decomp, self).__init__()
#        self.moving_avg = moving_avg(kernel_size, stride=1)
#
#    def forward(self, x):
#        moving_mean = self.moving_avg(x)
#        res = x - moving_mean
#        return res, moving_mean
#
#
#class ftblock(nn.Module):
#
#    def __init__(self, configs,affine = True, subtract_last = False):
#        super(ftblock, self).__init__()
#        self.pred_len = configs.pred_len
#        self.seq_len = configs.seq_len
#        self.lookback = configs.seq_len
#        self.stride = configs.stride
#        self.patch_size = configs.patch_len
#        self.d_model = configs.d_model
#        self.n_heads = configs.n_heads
#        self.d_inner = configs.d_model * configs.expand
#
#
#
#
#
#        #forecast1
#        self.hidden_size = 256
#        #self.decompsition = series_decomp(configs.kernel_size)
#        self.decompsition = series_decomp(configs.kernel_size)
#
#        self.n_vars = configs.enc_in
#        self.target_window = configs.pred_len
#
#        #趋势
#        self.regression_mlp = MLPRegression(self.seq_len, self.hidden_size, configs.pred_len)
#        #季节
#        self.patch_num = int((self.seq_len - self.patch_size) / self.stride + 1)
#        self.head_nf = self.d_model * self.patch_num
#        self.d_state = configs.d_state
#        self.d_conv = configs.d_conv
#        self.expand = configs.expand
#        
#        
#        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
#        self.backbone = Mamba_Backbone(self.patch_num, self.patch_size,self.n_vars,self.d_model,self.d_state,self.d_conv,self.expand,dropout=0.2)
#        self.head = Flatten_Head( self.n_vars, self.head_nf, self.target_window, head_dropout=0.2)
#        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,configs.dropout)
#        self.outlinear = nn.Linear(configs.pred_len+configs.pred_len,configs.pred_len)
#        self.dropout = nn.Dropout(p=0.2)
#        self.mamba = Mamba(
#            d_model=configs.d_model,
#            d_state=configs.d_state,
#            d_conv=configs.d_conv,
#            expand=configs.expand,
#        )
#        self.proj = nn.Linear(configs.d_model, configs.pred_len)
#
#
#    def forward(self, x_enc, x_mark_enc):
#        B, N, C = x_enc.shape
#
#        x_enc = self.revin_layer(x_enc, 'norm')
#        
#
#        enc_out = self.enc_embedding(x_enc, x_mark_enc)
#        enc_out = self.dropout(self.mamba(enc_out))
#        
#        
#        d_out = self.proj(enc_out).permute(0, 2, 1)[:, :, :C]
#
#       
#
#        # B,N,C
#        seasonal_init, trend_init = self.decompsition(x_enc)
#
#        # B,C,N
#        trend_init = trend_init.permute(0, 2, 1)
#        seasonal_init = seasonal_init.permute(0, 2, 1)
#        # seasonal_init = self.fre_pro(seasonal_init)
#
#       
#        trend_out = self.regression_mlp(trend_init)  # [B,E,pre_len]
#
#       
#        seasonal_patched = seasonal_init.unfold(dimension=-1, size=self.patch_size,step=self.stride)  # [B, C, patch_num, patch_len]
#       
#        seasonal_out = self.backbone(seasonal_patched, x_mark_enc)
#        seasonal_out = self.head(seasonal_out)  # [B,E,pre_len]
#        
#        out = seasonal_out + trend_out
#
#      
#
#        d_out = d_out.permute(0, 2, 1)
#
#        out = torch.cat((out, d_out), dim=-1)
#       
#
#        out = self.outlinear(out)
#        out = out.permute(0, 2, 1)
#
#        out = self.revin_layer(out, 'denorm')
#        return out
#
