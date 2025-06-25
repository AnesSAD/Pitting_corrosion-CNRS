import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
  
class UNET(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
      super(UNET, self).__init__()

      self.downs = nn.ModuleList()
      self.ups = nn.ModuleList()
      self.pool = nn.MaxPool2d(kernel_size=2,stride=2)


      # ENCODER
      for feature in features:
        self.downs.append(DoubleConv(in_channels,feature))
        in_channels = feature

      # DECODER
      for feature in features[::-1] :
        self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
        self.ups.append(DoubleConv(feature*2,feature))

      self.bottleneck = DoubleConv(features[-1],features[-1]*2)
      self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):

      skip_connections = []

      for down in self.downs:

        x = down(x)
        skip_connections.append(x)
        x= self.pool(x)
      x = self.bottleneck(x)

      skip_connections = skip_connections[::-1]

      for idx in range(0,len(self.ups),2):
        x = self.ups[idx](x)
        skip_connection = skip_connections[idx//2]

        if x.shape != skip_connection.shape:
          x = TF.resize(x,skip_connection.shape[2:])

        concat_skip = torch.cat((skip_connection,x),dim=1)

      x = self.ups[idx+1](concat_skip)

      return self.final_conv(x)

class DoubleConv(nn.Module):

  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, 3, 1, 1, bias=False), # 3 : convolutional core 3*3, 1 : no convolution, 1: border pixels
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self,x):
    return self.conv(x)

def create_model(config):
  ENCODER = config['encoder']
  ENCODER_WEIGHTS = config['weights']
  LR = config['lr']
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  loss = torch.nn.BCEWithLogitsLoss()
  model = smp.UnetPlusPlus(encoder_name=ENCODER,
                       encoder_weights = ENCODER_WEIGHTS,
                       classes = 1,
                       activiation = None,
                       in_channels = 1)

  model.to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(),lr=LR)

  return model, loss, optimizer
     

def test():

  x = torch.randn((1,1,161,161))
  model = UNET(in_channels=1,out_channels=1)
  preds = model(x)
  
  
 

