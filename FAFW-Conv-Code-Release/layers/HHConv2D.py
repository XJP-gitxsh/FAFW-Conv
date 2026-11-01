import torch.nn as nn

class HHconv2d(nn.Module):
    def __init__(self, nvars, drop=0.1):
        super(HHconv2d, self).__init__()


        self.norm = nn.BatchNorm2d(nvars)
        self.nvars = nvars

#        if nvars > 50 :
#            self.nvars = 1

        # convffn1
        self.ffn1pw1 = nn.Conv2d(
            in_channels=nvars ,
            out_channels=self.nvars,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1
        )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv2d(
            in_channels=self.nvars,
            out_channels=nvars,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1
        )
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)



    def forward(self, x):
        input = x


        B, M,num, D, N = x.shape

        x = x.reshape(B*num,M,N,D)
        x = self.norm(x)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        #x = x.reshape(B*num,M,D,N)
        x = self.ffn1drop2(self.ffn1pw2(x))
        


        
#        x_inverse = x.reshape(B * num, M, N, D)
#        x_inverse = self.norm(x_inverse)
#        x_inverse = self.ffn1drop1(self.ffn1pw1(x_inverse))
#        
#        x = x.reshape(B*num,M,D,N)
#        x = self.norm(x)
#        x = self.ffn1drop2(self.ffn1pw2(x))
#
#        x_inverse = x_inverse.reshape(B*num,M,D,N)
#        x = x + x_inverse
        

#        x = self.ffn1drop1(self.ffn1pw1(x))
#        x = self.ffn1act(x)
#        x = self.ffn1drop2(self.ffn1pw2(x))




        x = x.reshape(B, M, num,D, N)

        return x

