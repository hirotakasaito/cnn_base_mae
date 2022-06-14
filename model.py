import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import torchvision.models as models
import torchvision.transforms as transforms

class ResNetImageEncoder(nn.Module):
    """
    Encoder to embed camera observation to vector by ResNet18
    """
    def __init__(self, observation_size, embedding_size, hidden_size, dropout=0.5, return_img=False):
        super().__init__()

        self.dropout = dropout
        self.embedding_size = embedding_size

        resnet = models.resnet18(pretrained=True)
        self.resnet_layer = nn.Sequential(*list(resnet.children())[0:8])

        for param in self.resnet_layer.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 64, (1, 1), stride=(1,1)),
            nn.ReLU()
        )

        # convolution output size
        input_test = torch.randn(1, observation_size[0], observation_size[1], observation_size[2])
        conv_output = self.conv(self.resnet_layer(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)

        # self.fc1 = nn.Linear(self.conv_output_size, hidden_size)
        self.fc1 = nn.Linear(512, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, embedding_size)
        self.return_img = return_img

    def forward(self, obs):
        b, h, w, c = obs.size()
        _obs = obs.reshape([b,c,h,w])
        resnet_output = self.resnet_layer(_obs)
        resnet_output = F.adaptive_avg_pool2d(resnet_output,(1,1))
        x = resnet_output.reshape(b,-1)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.dropout > 0:
            x = self.dropout1(x)
        embedded_obs = F.relu(self.fc3(x)).reshape([b,self.embedding_size])

        return embedded_obs

class Vgg16ImageEncoder(nn.Module):
    """
    Encoder to embed camera observation to vector by vgg16
    """
    def __init__(self, observation_size, embedding_size, hidden_size, dropout=0.5, return_img=False):
        super().__init__()

        self.dropout = dropout
        self.embedding_size = embedding_size

        vgg16 = models.vgg16(pretrained=True)
        self.vgg16_layer = nn.Sequential(*list(vgg16.children())[:-2])

        for param in self.vgg16_layer.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 64, (1, 1), stride=(1,1)),
            nn.ReLU()
        )

        # convolution output size
        input_test = torch.randn(1, observation_size[0], observation_size[1], observation_size[2])
        conv_output = self.conv(self.vgg16_layer(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)

        self.fc1 = nn.Linear(self.conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, embedding_size)

        self.return_img = return_img


    def forward(self, obs):
        b, h, w,c = obs.size()
        _obs = obs.reshape([b,c,h,w])
        vgg16_output = self.vgg16_layer(_obs)
        conv_output = self.conv(vgg16_output)

        x = F.relu(self.fc1(conv_output.view(-1, self.conv_output_size)))
        x = F.relu(self.fc2(x))
        if self.dropout > 0:
            x = self.dropout1(x)
        embedded_obs = F.relu(self.fc3(x)).reshape([b,self.embedding_size])
        if(self.return_img):
            return embedded_obs, obs
        else:
            return embedded_obs

class ImageDecoder(nn.Module):

    def __init__(self, observation_size,embedded_obs,  embedding_size):
        super().__init__()

        # self.fc1 = nn.Linear(state_size+belief_size, embedding_size)
        self.fc2 = nn.Linear(embedded_obs, embedding_size)
        self.fc3 = nn.Linear(embedding_size, 64*8*8)

        # Deconv layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=4, padding=0), #(N,64,8,8) -> (N,32,32,32)
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), #(N,32,32,32) -> (N,16,64,64)
            nn.ELU(),
            nn.ConvTranspose2d(16, 3, 1, stride=1, padding=0), #(N,16,64,64) -> (N,3,256,256)
            # nn.ConvTranspose2d(16, 3, 2, stride=2, padding=0), #(N,16,64,64) -> (N,3,256,256)
            nn.Sigmoid()
        )

        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(64*8*8)

    def forward(self, embedded_obs):
        b,_ = embedded_obs.size()
        hidden = F.elu(self.fc2(embedded_obs))
        hidden = F.elu(self.fc3(hidden))
        deconv_input = hidden.view(-1, 64, 8, 8)
        observation = self.deconv(deconv_input)
        _, c, h, w = observation.size()
        observation = observation.reshape([b,h,w,c])
        return observation

class MultiAttentionNetwork(nn.Module):
    def __init__(self, observation_size,embedding_size,hidden_size, num_masks=4,dropout = 0.5,return_img=False):
        super().__init__()

        self.dropout = dropout
        self.embedding_size = embedding_size

        resnet = models.resnet18(pretrained=True)
        # vgg16 = models.vgg16(pretrained=True)
        # self.resnet_layer = nn.Sequential(*list(vgg16.children())[:-2])
        self.resnet_layer = nn.Sequential(*list(resnet.children())[0:8])

        for param in self.resnet_layer.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 64, (1, 1), stride=(1,1)),
            nn.ReLU()
        )

        # convolution output size
        input_test = torch.randn(1, observation_size[0], observation_size[1], observation_size[2])
        conv_output = self.conv(self.resnet_layer(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(512*num_masks,int(512*num_masks/2)),
            nn.ELU(),
            nn.Linear(int(512*num_masks/2),512),
            nn.ELU(),
            nn.Linear(512,embedding_size),
            nn.ELU(),
        )

        self.return_img = return_img

        self.attn_conv = nn.Conv2d(512, num_masks, 1, bias=False)

        self.mask_ = None
        self.num_masks = num_masks

    def forward(self, obs):
        b, h, w, c = obs.size()
        _obs = obs.reshape([b,c,h,w])
        resnet_output = self.resnet_layer(_obs)
        attn = torch.sigmoid(self.attn_conv(resnet_output))
        b, _, h, w = attn.shape
        self.mask_ = attn
        resnet_output = resnet_output.reshape(b, 1, 512, h, w)
        attn = attn.reshape(b,self.num_masks,1,h,w)
        resnet_output = resnet_output * attn
        resnet_output = resnet_output.reshape(b*self.num_masks,512,h,w)
        resnet_output = F.adaptive_avg_pool2d(resnet_output,(1,1))
        flatten = resnet_output.reshape(b,-1)

        if self.dropout > 0:
            x = self.dropout1(flatten)
        embedded_obs = self.fc(flatten)
        embedded_obs = embedded_obs.reshape(b,-1)
        if(self.return_img):
            return embedded_obs, obs
        else:
            return embedded_obs
class MultiAttentionNetworkWithVgg16(nn.Module):
    def __init__(self, observation_size,embedding_size,hidden_size, num_masks=4,dropout = 0.5,return_img=False):
        super().__init__()

        self.dropout = dropout
        self.embedding_size = embedding_size

        # resnet = models.resnet18(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16_layer = nn.Sequential(*list(vgg16.children())[:-2])
        # self.resnet_layer = nn.Sequential(*list(resnet.children())[0:8])

        for param in self.vgg16_layer.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 64, (1, 1), stride=(1,1)),
            nn.ReLU()
        )

        # convolution output size
        input_test = torch.randn(1, observation_size[0], observation_size[1], observation_size[2])
        conv_output = self.conv(self.vgg16_layer(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)

        self.fc = nn.Sequential(
            nn.Linear(512*num_masks,int(512*num_masks/2)),
            nn.ELU(),
            nn.Linear(int(512*num_masks/2),512),
            nn.ELU(),
            nn.Linear(512,embedding_size),
            nn.ELU(),
        )

        self.return_img = return_img

        self.attn_conv = nn.Conv2d(512, num_masks, 1, bias=False)

        self.mask_ = None
        self.num_masks = num_masks

    def forward(self, obs):
        b, h, w, c = obs.size()
        _obs = obs.reshape([b,c,h,w])
        vgg16_output = self.vgg16_layer(_obs)
        attn = torch.sigmoid(self.attn_conv(vgg16_output))
        b, _, h, w = attn.shape
        self.mask_ = attn
        vgg16_output = vgg16_output.reshape(b, 1, 512, h, w)
        attn = attn.reshape(b,self.num_masks,1,h,w)
        vattn = attn.clone()
        vgg16_output = vgg16_output * attn
        vgg16_output = vgg16_output.reshape(b*self.num_masks,512,h,w)
        vgg16_output = F.adaptive_avg_pool2d(vgg16_output,(1,1))
        flatten = vgg16_output.reshape(b,-1)

        if self.dropout > 0:
            x = self.dropout1(flatten)
        embedded_obs = self.fc(flatten)
        embedded_obs = embedded_obs.reshape(b,-1)
        if(self.return_img):
            return embedded_obs, obs
        else:
            return embedded_obs,vattn

