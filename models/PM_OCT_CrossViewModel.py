import torch.nn as nn
import torch
import torchvision
from models.FIT_Net import FITNet

def get_encoder(config, pretrained=False):
    encoder = {
        "FIT_Net": FITNet(pretrained=pretrained, num_classes=2, fusionFlag=config.fusionFlag),
    }
    if config.encoder not in encoder.keys():
        raise KeyError(f"{config.encoder} is not a valid model version")
    return encoder[config.encoder]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 2
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


# concatenate + MLP
class OCTCrossViewModel_CM(nn.Module):
    def __init__(self, encoder, n_features, config):
        super(OCTCrossViewModel_CM, self).__init__()
        self.config = config
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = config.projection_dim
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.n_features, config.classNumber)
        )

        self.projector = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features * 4, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features * 4, self.n_features, bias=False)
        )

    def forward(self, Himg, Vimg):
        hEncoderOut = self.encoder(Himg)
        vEncoderOut = self.encoder(Vimg)
        imgFeature = torch.cat((hEncoderOut, vEncoderOut), 1)
        imgFeature = self.projector(imgFeature)
        output = self.fc(imgFeature)
        return output

# concatenate
class OCTCrossViewModel_C(nn.Module):
    def __init__(self, encoder, n_features, config):
        super(OCTCrossViewModel_C, self).__init__()
        self.config = config
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = config.projection_dim
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.n_features * 2, config.classNumber)
        )

    def forward(self, Himg, Vimg):
        hEncoderOut = self.encoder(Himg)
        vEncoderOut = self.encoder(Vimg)
        imgFeature = torch.cat((hEncoderOut, vEncoderOut), 1)
        output = self.fc(imgFeature)
        return output

# point-wise addition
class OCTCrossViewModel_A(nn.Module):
    def __init__(self, encoder, n_features, config):
        super(OCTCrossViewModel_A, self).__init__()
        self.config = config
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = config.projection_dim
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.n_features, config.classNumber)
        )

    def forward(self, Himg, Vimg):
        hEncoderOut = self.encoder(Himg)
        vEncoderOut = self.encoder(Vimg)
        imgFeature = hEncoderOut + vEncoderOut
        output = self.fc(imgFeature)
        return output

# point-wise addition + MLP
class OCTCrossViewModel_AM(nn.Module):
    def __init__(self, encoder, n_features, config):
        super(OCTCrossViewModel_AM, self).__init__()
        self.config = config
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = config.projection_dim
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.n_features, config.classNumber)
        )

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 2, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features * 2, self.n_features, bias=False)
        )

    def forward(self, Himg, Vimg):
        hEncoderOut = self.encoder(Himg)
        vEncoderOut = self.encoder(Vimg)
        imgFeature = hEncoderOut + vEncoderOut
        imgFeature = self.projector(imgFeature)
        output = self.fc(imgFeature)
        return output

# deep concatenate + MLP
class OCTCrossViewModel_DCM(nn.Module):
    def __init__(self, encoder, n_features, config):
        super(OCTCrossViewModel_DCM, self).__init__()
        self.dim = [1536, 384, 2048]
        self.config = config
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = config.projection_dim
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.dim[0] + self.dim[1] + self.dim[2], config.classNumber)
        )

        self.MLP1 = Mlp(in_features=self.dim[0] * 2, out_features=self.dim[0])
        self.MLP2 = Mlp(in_features=self.dim[1] * 2, out_features=self.dim[1])
        self.MLP3 = Mlp(in_features=self.dim[2] * 2, out_features=self.dim[2])

    def forward(self, Himg, Vimg):
        hEncoderOut, hClassToken1, hClassToken2, hClassToken3 = self.encoder(Himg)
        vEncoderOut, vClassToken1, vClassToken2, vClassToken3 = self.encoder(Vimg)
        imgFeature_1 = torch.cat((hClassToken1, vClassToken1), 1)
        imgFeature_1 = self.MLP1(imgFeature_1)

        imgFeature_2 = torch.cat((hClassToken2, vClassToken2), 1)
        imgFeature_2 = self.MLP2(imgFeature_2)

        imgFeature_3 = torch.cat((hClassToken3, vClassToken3), 1)
        imgFeature_3 = self.MLP3(imgFeature_3)

        imgFeature = torch.cat((imgFeature_1, imgFeature_2, imgFeature_3), 1)
        output = self.fc(imgFeature)
        return output

# deep point-wise addition + MLP
class OCTCrossViewModel_DAM(nn.Module):
    def __init__(self, encoder, n_features, config):
        super(OCTCrossViewModel_DAM, self).__init__()
        self.dim = [1536, 384, 2048]
        self.config = config
        self.encoder = encoder
        self.n_features = n_features
        self.projection_dim = config.projection_dim
        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(self.dim[0] + self.dim[1] + self.dim[2], config.classNumber)
        )

        self.MLP1 = Mlp(in_features=self.dim[0])
        self.MLP2 = Mlp(in_features=self.dim[1])
        self.MLP3 = Mlp(in_features=self.dim[2])

    def forward(self, Himg, Vimg):
        hEncoderOut, hClassToken1, hClassToken2, hClassToken3 = self.encoder(Himg)
        vEncoderOut, vClassToken1, vClassToken2, vClassToken3 = self.encoder(Vimg)
        imgFeature_1 = hClassToken1 + vClassToken1
        imgFeature_1 = self.MLP1(imgFeature_1)

        imgFeature_2 = hClassToken2 + vClassToken2
        imgFeature_2 = self.MLP2(imgFeature_2)

        imgFeature_3 = hClassToken3 + vClassToken3
        imgFeature_3 = self.MLP3(imgFeature_3)

        imgFeature = torch.cat((imgFeature_1, imgFeature_2, imgFeature_3), 1)
        output = self.fc(imgFeature)
        return output

if __name__ == '__main__':
    import torch

    class opt:
        projection_dim = 768
        encoder = 'FIT_Net'
        classNumber = 2
        preWeight = None
        fusionFlag = True

    config = opt

    encoder = get_encoder(config, pretrained=False)
    #  head, fc get dimensions of fc layer
    n_features = encoder.fc.in_features
    model = OCTCrossViewModel_DAM(encoder, n_features, config)
    Himg = torch.randn(1, 3, 224, 224)
    Vimg = torch.randn(1, 3, 224, 224)
    output = model(Himg, Vimg)
    print('output shape:{}'.format(output.shape))
