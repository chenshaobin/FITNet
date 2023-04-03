import torch.nn as nn
import torch
import torchvision.models as models_torch
from models.DST_block_IA import DST_block_IA

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Mlp(nn.Module):

    def __init__(self, in_features, out_feature, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_feature)
        # self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act(x)
        x = self.drop(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, fusionFlag=False):
        super(ResNet, self).__init__()
        self.fusionFlag = fusionFlag
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   # 112 X 112
        self.scale1Conv1 = nn.Conv2d(64, 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)     # 56 X 56

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.scale2Conv1 = nn.Conv2d(256, 8, kernel_size=1, bias=False)
        self.DST_block_IA_stage1 = DST_block_IA(
            num_classes=2, depth=2,
            sm_image_size=112, sm_channel=4, sm_dim=1024, sm_patch_size=16, sm_enc_depth=3, sm_enc_heads=4,
            sm_enc_mlp_dim=2048,
            lg_image_size=56, lg_channel=8, lg_dim=512, lg_patch_size=8, lg_enc_depth=3, lg_enc_heads=4,
            lg_enc_mlp_dim=1024,
            cross_attn_depth=2, cross_attn_heads=4, dropout=0.1, emb_dropout=0.1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.scale3Conv1 = nn.Conv2d(512, 16, kernel_size=1, bias=False)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.scale4Conv1 = nn.Conv2d(1024, 32, kernel_size=1, bias=False)

        self.DST_block_IA_stage2 = DST_block_IA(
            num_classes=2, depth=2,
            sm_image_size=28, sm_channel=16, sm_dim=256, sm_patch_size=4, sm_enc_depth=3, sm_enc_heads=4,
            sm_enc_mlp_dim=512,
            lg_image_size=14, lg_channel=32, lg_dim=128, lg_patch_size=2, lg_enc_depth=3, lg_enc_heads=4,
            lg_enc_mlp_dim=256,
            cross_attn_depth=2, cross_attn_heads=4, dropout=0.1, emb_dropout=0.1)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = Mlp(in_features=3968, out_feature=1984, act_layer=nn.GELU, drop=0.4)
        self.fc = nn.Linear(1984, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = self.conv1(x)
        scale1 = self.scale1Conv1(x_1)
        x = self.bn1(x_1)
        x = self.relu(x)
        x = self.maxpool(x)
        res1 = self.layer1(x)
        scale2 = self.scale2Conv1(res1)
        cls_token1 = self.DST_block_IA_stage1(scale1, scale2)

        res2 = self.layer2(res1)
        scale3 = self.scale3Conv1(res2)

        res3 = self.layer3(res2)
        scale4 = self.scale4Conv1(res3)

        cls_token2 = self.DST_block_IA_stage2(scale3, scale4)

        res4 = self.layer4(res3)
        x = self.avgpool(res4)
        cls_token3 = x.view(x.size(0), -1)

        out = torch.cat((cls_token1, cls_token2, cls_token3), dim=-1)
        out = self.mlp(out)
        out = self.fc(out)
        if not self.fusionFlag:
            return out
        else:
            return out, cls_token1, cls_token2, cls_token3


def FITNet(pretrained=True, **kwargs):
    """
    Constructs a FIT-Net model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = models_torch.resnet50(pretrained=pretrained)
        pretrained_dict = pretrained_model.state_dict()
        new_params = model.state_dict().copy()

        for name, param in new_params.items():
            if name in pretrained_dict and param.size() == pretrained_dict[name].size():
                new_params[name].copy_(pretrained_dict[name])
        model.load_state_dict(new_params)

    return model


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    model = FITNet()
    model.fc = torch.nn.Linear(model.fc.in_features, 4)

    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)

    pre = model(img)
    print(pre.shape)

