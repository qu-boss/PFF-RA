1.模型修改后的三个网络结构文件分别是：yolov7-tiny-silu_all_att.yaml、yolov7_all_att.yaml和yolov7x_all_att.yaml。这些文件都在cfg/training目录下
2.本文所创新的模块都在models/common.py文件中，其中
    class RA_1(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(RA_1, self).__init__()
            self.convert = nn.Conv2d(in_channel, out_channel, 1)
            # self.convert1 = nn.Conv2d(256,128,1)  #yolov7_tiny_silu
            self.convert1 = nn.Conv2d(512,256,1)  #yolov7
            # self.convert1 = nn.Conv2d(640,320,1)  #yolov7x
            self.bn = nn.BatchNorm2d(out_channel)
            self.silu = nn.SiLU(True)
            self.convs = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.SiLU(True),
                nn.Conv2d(out_channel, out_channel, 1), nn.BatchNorm2d(out_channel), nn.SiLU(True),
            )
            self.channel = out_channel
    
        def forward(self, input):
            x = input[0]
            x_size = x.size()[2:]
            y = input[1]
            y1 = F.interpolate(y, x_size, mode='bilinear', align_corners=True)
            y1 = self.bn(self.convert1(y1))
            a = torch.sigmoid(-y1)
            x = x.mul(a)
            out = y1 + self.convs(x)
            return out
    
        def initialize(self):
            weight_init(self)
