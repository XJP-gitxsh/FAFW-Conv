import torch
from torch import nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from models.ModernTCN_Layer import series_decomp, Flatten_Head
import pywt
from layers.WTconv import WTConv2d,create_wavelet_filter,inverse_wavelet_transform,wavelet_transform
from layers.HHConv2D import HHconv2d
#from layers.Ftblock import ftblock5

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFractionalConv1d(nn.Conv1d):
    default_fractional_mode = "difference"
    """
    Adaptive Fractional Memory Convolution

    Core:
        X --> Fractional Memory --> Conv

    Features
    --------
    ✓ Learnable alpha
    ✓ Dynamic alpha
    ✓ Dynamic lambda
    ✓ Residual fusion
    ✓ Variance alignment
    ✓ Compatible with ModernTCN
    ✓ Compatible with TimesNet
    ✓ Compatible with TCN
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,

            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,

            alpha_init=0.3,
            reduction=16,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.fractional_mode = self.default_fractional_mode

        # In the ordinary-convolution control, do not create unused fractional
        # parameters; forward returns the untouched Conv1d result.
        if self.fractional_mode == "none":
            return

        if isinstance(kernel_size, tuple):
            self.k = kernel_size[0]
        else:
            self.k = kernel_size

        hidden = max(in_channels // reduction, 4)

        # ===================================
        # Dynamic alpha generator
        # ===================================

        self.alpha_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1)
        )

        # initialize

        self.alpha_bias = nn.Parameter(
            torch.tensor(alpha_init)
        )

        # ===================================
        # Dynamic lambda
        # ===================================

        self.lambda_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1)
        )

        # ===================================
        # Residual gate
        # ===================================

        self.residual_gate = nn.Parameter(
            torch.tensor(0.2)
        )

        # ===================================
        # Temperature
        # ===================================

        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )

    # ====================================================
    # G-L kernel
    # ====================================================

    def build_fractional_kernel(self, alpha):

        coeffs = []

        c = torch.ones(
            alpha.shape[0],
            device=alpha.device
        )

        for i in range(self.k):

            if i == 0:
                coeffs.append(c)

            else:

                if self.fractional_mode == "sum":
                    c = c * (alpha + (i - 1)) / i
                    coeffs.append(c)
                else:
                    c = c * (alpha - (i - 1)) / i
                    coeffs.append(((-1) ** i) * c)

        coeffs = torch.stack(coeffs, dim=-1)

        coeffs = coeffs / (
                coeffs.abs().sum(-1, keepdim=True)
                + 1e-6
        )

        coeffs = coeffs * self.temperature

        return coeffs

    # ====================================================
    # Fractional Memory
    # ====================================================

    def fractional_memory(self, x, frac_kernel):

        B, C, T = x.shape

        outputs = []

        for b in range(B):

            kernel = frac_kernel[b]

            kernel = kernel.view(1, 1, self.k)

            kernel = kernel.repeat(C, 1, 1)

            # PyTorch conv1d is cross-correlation. Flip lag-ordered weights
            # and pad only on the left to implement a causal history filter.
            out = F.conv1d(
                F.pad(x[b:b + 1], (self.k - 1, 0)),
                kernel.flip(-1),
                groups=C
            )

            outputs.append(out)

        return torch.cat(outputs, dim=0)

    # ====================================================
    # Forward
    # ====================================================

    def forward(self, x):

        if self.fractional_mode == "none":
            return F.conv1d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups
            )

        # ===================================
        # Dynamic alpha
        # ===================================

        alpha = self.alpha_generator(x)

        alpha = torch.sigmoid(
            alpha.squeeze(-1).squeeze(-1)
            + self.alpha_bias
        )

        # ===================================
        # Dynamic lambda
        # ===================================

        frac_strength = torch.sigmoid(
            self.lambda_generator(x)
        )

        frac_strength = frac_strength.squeeze(-1).squeeze(-1)

        # ===================================
        # Build G-L kernel
        # ===================================

        frac_kernel = self.build_fractional_kernel(alpha)

        # ===================================
        # Fractional memory branch
        # ===================================

        x_frac = self.fractional_memory(
            x,
            frac_kernel
        )

        # ===================================
        # Strength modulation
        # ===================================

        strength = frac_strength.view(
            -1, 1, 1
        )

        x_frac = (
                (1 - strength) * x
                + strength * x_frac
        )

        # ===================================
        # Standard branch
        # ===================================

        base_out = F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        # ===================================
        # Fractional branch
        # ===================================

        frac_out = F.conv1d(
            x_frac,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        # ===================================
        # Variance alignment
        # ===================================

        base_std = (
                base_out.std(
                    dim=-1,
                    keepdim=True
                ).detach()
                + 1e-6
        )

        frac_std = (
                frac_out.std(
                    dim=-1,
                    keepdim=True
                ).detach()
                + 1e-6
        )

        frac_out = frac_out * (
                base_std / frac_std
        )

        # ===================================
        # Residual fusion
        # ===================================

        gate = torch.sigmoid(
            self.residual_gate
        )

        out = (
                (1 - gate) * base_out
                + gate * frac_out
        )

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class FractionalConv1d(nn.Conv1d):
    # Set from Model(configs) before constructing the backbone.
    default_fractional_mode = "difference"
    default_use_kernel1_modulation = True
    """
    Fractional-order Conv1d
    =======================

    Direct replacement for nn.Conv1d

    Features
    --------
    ✓ Fully compatible with:
        - groups
        - dilation
        - padding
        - depthwise conv
        - residual architectures
        - ModernTCN
        - TimesNet
        - PatchTST
        - TCN

    ✓ Stable training
    ✓ Learnable fractional order when kernel_size > 1
    ✓ Finite-kernel weight modulation
    ✓ Lightweight implementation
    ✓ No dimension mismatch

    --------------------------------------------------

    Core Idea
    ----------
    Standard Conv1d:
        y = W * x

    Fractional Conv1d:
        W_frac = W ⊙ (1 + beta * G_alpha)
        y_frac = Conv(x, W_frac, bias=None)
        y = (1 - sigmoid(rho)) * Conv(x, W, bias) + sigmoid(rho) * AlignStd(y_frac)

    where G_alpha is a truncated, L1-normalized Grünwald–Letnikov kernel,
    beta is learnable, rho is a learnable scalar gate, and AlignStd rescales
    the fractional branch to the standard branch's per-sample/channel temporal
    standard deviation. This is a weight modulation, not input-side filtering.

    This introduces:
        - a finite-kernel weight transform when kernel_size > 1

    Important for the current ModernTCN backbone:
        This class is instantiated only with kernel_size=1. Its GL kernel
        therefore has one coefficient and cannot mix temporal lags. In that
        use it has no lag information. The fractional weight scaling is
        canceled by standard-deviation alignment for a positive one-tap
        scale; because the fractional branch omits bias, the gate mainly
        mixes a biased and a bias-free pointwise convolution. It is not a
        temporal-memory or power-law filter.

    --------------------------------------------------

    In the current ModernTCN backbone, this class is instantiated with
    kernel_size=1. The coefficient then contains no lag information, alpha
    has no effect, and standard-deviation alignment cancels positive scalar
    weight modulation. The gate blends the ordinary biased output with the
    aligned bias-free pointwise output.

    --------------------------------------------------

    Example
    -------
    Replace directly:

        nn.Conv1d(...)  ->
        FractionalConv1d(...)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,

        # fractional params
        alpha_init=0.3,
        learnable_alpha=True,

        # fractional modulation strength
        frac_strength_init=0.1,

        # stability
        residual_gate_init=0.2,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # kernel size
        if isinstance(kernel_size, tuple):
            self.k = kernel_size[0]
        else:
            self.k = kernel_size

        self.fractional_mode = self.default_fractional_mode
        self.use_kernel1_modulation = self.default_use_kernel1_modulation

        if self.fractional_mode == "none" or (
            self.k == 1 and not self.use_kernel1_modulation
        ):
            return

        # ==========================================
        # Learnable fractional order α
        # ==========================================

        if learnable_alpha:
            self.alpha_raw = nn.Parameter(
                torch.tensor(float(alpha_init))
            )
        else:
            self.register_buffer(
                "alpha_raw",
                torch.tensor(float(alpha_init))
            )

        # ==========================================
        # Learnable fractional modulation strength
        # ==========================================

        self.frac_strength = nn.Parameter(
            torch.tensor(float(frac_strength_init))
        )

        # ==========================================
        # Residual gate
        # ==========================================

        self.residual_gate = nn.Parameter(
            torch.tensor(float(residual_gate_init))
        )

        # ==========================================
        # Learnable temperature
        # improves adaptability
        # ==========================================

        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )

    # =========================================================
    # Build fractional kernel
    # =========================================================

    def build_fractional_kernel(self):

        # -----------------------------------------
        # bound α into stable range
        # α ∈ (0, 1)
        # -----------------------------------------

        alpha = torch.sigmoid(self.alpha_raw)

        coeffs = []

        c = torch.ones(
            1,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        for i in range(self.k):

            if i == 0:
                coeffs.append(c)

            else:

                c = c * (alpha - (i - 1)) / i

                coeffs.append(
                    ((-1) ** i) * c
                )

        coeffs = torch.cat(coeffs)

        # =====================================================
        # IMPORTANT:
        # Normalize for numerical stability
        # =====================================================

        coeffs = coeffs / (
            coeffs.abs().sum() + 1e-6
        )

        # ==========================================
        # Temperature scaling
        # ==========================================

        coeffs = coeffs * self.temperature

        return coeffs.view(1, 1, -1)

    # =========================================================
    # Forward
    # =========================================================

    def forward(self, x):

        # ==========================================
        # Standard convolution branch
        # ==========================================

        base_out = F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if self.fractional_mode == "none" or (
            self.k == 1 and not self.use_kernel1_modulation
        ):
            return base_out

        # ==========================================
        # Fractional kernel
        # ==========================================

        frac_kernel = self.build_fractional_kernel()

        # ==========================================
        # Fractional modulation
        # ==========================================

        frac_weight = self.weight * (
            1.0
            + self.frac_strength * frac_kernel
        )

        # ==========================================
        # Fractional convolution branch
        # ==========================================

        frac_out = F.conv1d(
            x,
            frac_weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # =====================================================
        # VERY IMPORTANT:
        # Variance alignment
        # Prevent exploding/vanishing
        # =====================================================

        base_std = base_out.std(
            dim=-1,
            keepdim=True
        ).detach()

        frac_std = frac_out.std(
            dim=-1,
            keepdim=True
        ).detach() + 1e-6

        frac_out = frac_out * (
            base_std / frac_std
        )

        # =====================================================
        # Residual fusion
        # =====================================================

        gate = torch.sigmoid(
            self.residual_gate
        )

        out = (
            (1.0 - gate) * base_out
            + gate * frac_out
        )

        return out




import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedFractionalConv1d(nn.Conv1d):

    # Set once from Model(configs) before constructing the network.
    default_fractional_mode = "difference"
    """
    Fixed Fractional Memory Convolution

    alpha is fixed and not learnable.

    Used for:
        Ablation:
        Fixed alpha vs Adaptive alpha

    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,

            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,

            alpha=0.3,
            frac_strength=0.5,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.fractional_mode = self.default_fractional_mode


        if isinstance(kernel_size, tuple):
            self.k = kernel_size[0]
        else:
            self.k = kernel_size


        # =====================================
        # Fixed fractional order
        # =====================================

        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha))
        )


        # =====================================
        # Fixed fractional strength
        # =====================================

        self.register_buffer(
            "frac_strength",
            torch.tensor(float(frac_strength))
        )

        if self.fractional_mode == "none":
            return


        # residual fusion
        self.residual_gate = nn.Parameter(
            torch.tensor(0.2)
        )


        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )



    # =====================================
    # Generate GL fractional kernel
    # =====================================

    def build_fractional_kernel(self):

        alpha = self.alpha


        coeffs = []

        c = torch.ones(
            1,
            device=self.weight.device,
            dtype=self.weight.dtype
        )


        mode = self.fractional_mode
        for i in range(self.k):

            if i == 0:

                coeffs.append(c)


            else:

                if mode == "sum":
                    # (1-L)^(-alpha): causal fractional accumulation.
                    c = c * (alpha + (i-1)) / i
                    coeffs.append(c)
                else:
                    # (1-L)^alpha: fractional differencing.
                    c = c * (alpha - (i-1)) / i
                    coeffs.append((-1)**i * c)


        coeffs = torch.cat(coeffs)


        # normalization

        coeffs = coeffs / (
            coeffs.abs().sum()
            + 1e-6
        )


        coeffs = coeffs * self.temperature


        return coeffs.view(
            1,1,-1
        )



    # =====================================
    # Fractional memory operation
    # =====================================

    def fractional_memory(
            self,
            x,
            kernel
    ):

        B,C,T=x.shape


        kernel = kernel.repeat(
            C,
            1,
            1
        )


        # kernel entries are ordered by lag: [w_0, w_1, ...]. Conv1d uses
        # cross-correlation, so flip them and apply left-only causal padding.
        return F.conv1d(
            F.pad(x, (self.k - 1, 0)),
            kernel.flip(-1),
            groups=C
        )



    # =====================================
    # Forward
    # =====================================

    def forward(self,x):


        # -------------------------------
        # Standard convolution
        # -------------------------------

        base_out = F.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        # A one-tap normalized G-L kernel is the identity and carries no
        # temporal lag information. Avoid computing an identical second path.
        if self.fractional_mode == "none" or self.k == 1:
            return base_out



        # -------------------------------
        # Fractional memory
        # -------------------------------

        frac_kernel = (
            self.build_fractional_kernel()
        )


        x_frac = self.fractional_memory(
            x,
            frac_kernel
        )



        strength = self.frac_strength


        x_frac = (
            (1-strength)*x
            +
            strength*x_frac
        )



        # -------------------------------
        # Fractional convolution
        # -------------------------------

        frac_out = F.conv1d(
            x_frac,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


        # -------------------------------
        # Variance alignment
        # -------------------------------

        base_std = (
            base_out.std(
                dim=-1,
                keepdim=True
            ).detach()
            +1e-6
        )


        frac_std = (
            frac_out.std(
                dim=-1,
                keepdim=True
            ).detach()
            +1e-6
        )


        frac_out = frac_out * (
            base_std / frac_std
        )



        # -------------------------------
        # Residual fusion
        # -------------------------------

        gate=torch.sigmoid(
            self.residual_gate
        )


        out = (
            (1-gate)*base_out
            +
            gate*frac_out
        )


        return out












class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return FixedFractionalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = FixedFractionalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)

        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self):

        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)

        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)

            eq_b += small_b

            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = FixedFractionalConv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class Block(nn.Module):
    default_use_hhconv = True
    default_use_wavelet = True

    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()

        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        #convffn1
        self.ffn1pw1 = FixedFractionalConv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = FixedFractionalConv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = FixedFractionalConv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = FractionalConv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel

        self.use_wavelet = self.default_use_wavelet
        self.use_hhconv = self.default_use_hhconv
        if self.use_wavelet:
            self.wt_filter, self.iwt_filter = create_wavelet_filter('db1', nvars, nvars, torch.float)
            self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
            self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
            self.HHconv2d = HHconv2d(nvars,drop=drop) if self.use_hhconv else nn.Identity()
        else:
            self.HHconv2d = nn.Identity()



    def forward(self,x):

        input = x
        # B, M, D, N = x.shape
        u,v,w,num = input.shape
        

        

        if self.use_wavelet:
            x_wt = wavelet_transform(x,self.wt_filter)
            x = x_wt[:, :, 0, :, :]
            B, M, D, N = x.shape
            x_hh =x_wt[:, :, 1:4, :, :]
            if self.use_hhconv:
                x_hh = self.HHconv2d(x_hh)
        else:
            B, M, D, N = x.shape
            x_hh = None



        x = x.reshape(B,M*D,N)
        x = self.dw(x)
        x = x.reshape(B,M,D,N)
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

#        x = x.unsqueeze(2)
#        x = x.repeat(1, 1, 4, 1, 1)
	
        if self.use_wavelet:
            x = torch.cat([x.unsqueeze(2), x_hh], dim=2)
            x = inverse_wavelet_transform(x, self.iwt_filter)
            if num%2 ==1:
                pad = x[:, :, :, -1:].repeat(1, 1, 1, 1)
                x = torch.cat([x, pad], dim=-1)


        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        block_dmodel = dmodel // 2 if Block.default_use_wavelet else dmodel
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=block_dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x




class ModernTCN(nn.Module):
    def __init__(self,patch_size,patch_stride, stem_ratio, downsample_ratio, ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True, affine=True,
                 subtract_last=False, freq=None, seq_len=512, c_in=7, individual=False, target_window=96):

        super(ModernTCN, self).__init__()



        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers(if needed)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(

            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                FixedFractionalConv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        if freq == 'h':
            time_feature_num = 4
        elif freq == 't':
            time_feature_num = 5
        else:
            raise NotImplementedError("time_feature_num should be 4 or 5")

        self.te_patch = nn.Sequential(

            FixedFractionalConv1d(time_feature_num, time_feature_num, kernel_size=patch_size, stride=patch_stride,groups=time_feature_num),
            FractionalConv1d(time_feature_num, dims[0], kernel_size=1, stride=1, groups=1),
            nn.BatchNorm1d(dims[0]))

        # backbone

        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx], dmodel=dims[stage_idx],
                          dw_model=dw_dims[stage_idx], nvars=nvars, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)

        # Multi scale fusing (if needed)
        self.use_multi_scale = use_multi_scale
        self.up_sample_ratio = downsample_ratio

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        for i in range(self.num_stage):
            align_dim = dims[-1]
            lat = FixedFractionalConv1d(dims[i], align_dim, kernel_size=1,
                            stride=1)
            self.lat_layer.append(lat)
            smooth = FixedFractionalConv1d(align_dim, align_dim, kernel_size=3, stride=1, padding=1)
            self.smooth_layer.append(smooth)

            up_conv = nn.Sequential(
                nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.up_sample_ratio, stride=self.up_sample_ratio),
                nn.BatchNorm1d(align_dim))
            self.up_sample_conv.append(up_conv)

        # head
        patch_num = seq_len // patch_stride

        self.n_vars = c_in
        self.individual = individual
        d_model = dims[-1]
        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        else:

            if patch_num % pow(downsample_ratio,(self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio,(self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1))+1)
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)


    def up_sample(self, x, upsample_ratio):
        _, _, _, N = x.shape
        return F.upsample(x, size=N, scale_factor=upsample_ratio, mode='bilinear')



    def forward_feature(self, x, te=None):

        B,M,L=x.shape

        x = x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape


            x = x.reshape(B * M, D, N)
            if i==0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:,:,-1:].repeat(1,1,pad_len)
                    x = torch.cat([x,pad],dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]],dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, x, te=None):

        # instance norm
        if self.revin:
            x = x.permute(0, 2, 1)#B,L,N
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)#B,N,L
        x = self.forward_feature(x,te)
        x = self.head(x)
        # de-instance norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0, 2, 1)
        return x

    def encode_features(self, x, te=None):
        """Encode [B, time, variables] into pre-head features."""
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)
        return self.forward_feature(x, te)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        FixedFractionalConv1d.default_fractional_mode = getattr(
            configs, "fractional_mode", "difference"
        )
        AdaptiveFractionalConv1d.default_fractional_mode = FixedFractionalConv1d.default_fractional_mode
        FractionalConv1d.default_fractional_mode = FixedFractionalConv1d.default_fractional_mode
        FractionalConv1d.default_use_kernel1_modulation = getattr(
            configs, "use_kernel1_modulation", True
        )
        Block.default_use_hhconv = getattr(configs, "use_hhconv", True)
        Block.default_use_wavelet = getattr(configs, "use_wavelet", True)
        # hyper param

        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = self.nvars,
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
#
#        self.ft_block = ftblock(configs)



        # decomp
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_res = ModernTCN(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
            self.model_trend = ModernTCN(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
        else:
            self.model = ModernTCN(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)

    def forward(self, x, te=None):
#        x_ft = self.ft_block(x,te)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            res = self.model_res(res_init, te)
            trend = self.model_trend(trend_init, te)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            x = self.model(x, te)
            x = x.permute(0, 2, 1)
#        x = x + x_ft
        return x
