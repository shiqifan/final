import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange , Reduce

class PatchEmbedding(nn.Module):
    def __init__(self , in_channels : int = 3,patch_size : int = 16 , emb_size : int = 768 , img_size = 224) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels,emb_size,kernel_size=self.patch_size,stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
    
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1 , emb_size))
    def forward(self , x:torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.projection(x)
        cls_tockens = repeat(self.cls_token,'() n e -> b n e',b = b) 
        
        x = torch.cat([cls_tockens , x] , dim=1)
        
        x += self.positions
        return x
        

class ClassficationHead(nn.Module):
    def __init__(self,emb_size = 768 , n_classes = 100) -> None:
        super().__init__()
        
        self.seq = nn.Sequential(
            Reduce('b n e -> b e',reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size , n_classes),
            
        )
    def forward(self,x):
        
        return self.seq(x)
        


class Vit(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 patch_size = 16,
                 img_size   = 224,
                 d_model:int = 768,n_head:int = 8 , dim_feed_foward:int = 2048 , num_layers:int = 6,n_classes = 100) -> None:
        super().__init__()
        
        self.tlayer = nn.TransformerEncoderLayer(d_model,n_head,dim_feedforward=dim_feed_foward,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.tlayer,num_layers)
        self.embedding = PatchEmbedding(in_channels , patch_size , emb_size = d_model,img_size = img_size)
        self.cls_head = ClassficationHead(d_model,n_classes)
        
    def forward(self,x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.cls_head(x)

# load resnet18 from torchvision
def load_resnet18():
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    return model
# summary of the model ,how many parameters are in the model
def summary(model):
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

#summary the model use torchsummary
def summary_torch(model,input_size):
    from torchsummary import summary
    summary(model,input_size=(3,input_size,input_size),device='cpu')
 
if __name__ == '__main__':
    summary_torch(load_resnet18(),input_size=224)
    summary(Vit(in_channels=3,patch_size=16,img_size=32,d_model=256,n_head=8,dim_feed_foward=2048,num_layers=8,n_classes=100))
    