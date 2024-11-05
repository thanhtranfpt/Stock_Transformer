import lightning as L
# import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


def generate_pseudo_masks(ts_size: int, num_samples: int):

    # Tạo mask với toàn bộ giá trị False
    masks = torch.zeros((num_samples, ts_size), dtype=torch.bool)

    return masks


def generate_random_masks(num_samples: int, ts_size: int, mask_size: int, num_masks: int):
    num_patches = int(ts_size // mask_size)  # Số lượng patch

    def single_sample_mask():
        idx = torch.randperm(num_patches)[:num_masks]  # Lựa chọn ngẫu nhiên các patch
        mask = torch.zeros(ts_size, dtype=torch.bool)
        for j in idx:
            mask[j * mask_size:(j + 1) * mask_size] = 1  # Gán mask vào các patch
        return mask

    # Tạo danh sách mask cho mỗi sample
    masks_list = [single_sample_mask() for _ in range(num_samples)]
    
    # Stack các mask thành tensor với shape (num_samples, ts_size)
    masks = torch.stack(masks_list, dim=0)  
    
    return masks


def mask_it(x: torch.Tensor, masks: torch.Tensor):
    """
    Args:
        x.shape = (batch_size, ts_size, f)
        masks.shape = (batch_size, ts_size), với mỗi giá trị là True hoặc False (True nghĩa là bị mask)

    """

    b, l, f = x.shape  # b: batch_size, l: ts_size, f: f
    
    # Đảm bảo masks có shape là (batch_size, ts_size)
    assert masks.shape == (b, l), "Shape của masks phải là (batch_size, ts_size)"
    
    # Mở rộng mask sang f (feature dimension) để khớp với x
    masks_expanded = masks.unsqueeze(-1).expand(-1, -1, f)  # (batch_size, ts_size, f)
    
    # Chỉ giữ lại các phần tử không bị mask (masks == 0)
    x_visible = x[~masks_expanded].reshape(b, -1, f)  # (batch_size, vis_size, f)
    
    return x_visible


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict):     {
                                'z_dim': int = 6,  # Number of Features
                                'hidden_dim': int = 12,
                                'num_layers': int = 3
                            }

        """

        super().__init__()

        self.rnn = nn.RNN(input_size=cfg['z_dim'],
                          hidden_size=cfg['hidden_dim'],
                          num_layers=cfg['num_layers'])
        
        self.fc = nn.Linear(in_features=cfg['hidden_dim'],
                            out_features=cfg['hidden_dim'])

    def forward(self, x):

        x_enc, _ = self.rnn(x)

        x_enc = self.fc(x_enc)

        return x_enc
    

class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict):     {
                                'z_dim': int = 6,  # Number of Features
                                'hidden_dim': int = 12,
                                'num_layers': int = 3
                            }

        """

        super().__init__()

        self.rnn = nn.RNN(input_size=cfg['hidden_dim'],
                          hidden_size=cfg['hidden_dim'],
                          num_layers=cfg['num_layers'])
        
        self.fc = nn.Linear(in_features=cfg['hidden_dim'],
                            out_features=cfg['z_dim'])

    def forward(self, x_enc):

        x_dec, _ = self.rnn(x_enc)

        x_dec = self.fc(x_dec)

        return x_dec
    

class Interpolator(nn.Module):
    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict):     {
                                'ts_size': int = 24  # Time-Series size,
                                'total_mask_size': int = 3,
                                'hidden_dim': int = 12,
                            }

        """

        super().__init__()

        self.sequence_inter = nn.Linear(in_features=(cfg['ts_size'] - cfg['total_mask_size']),
                                        out_features=cfg['ts_size'])
        
        self.feature_inter = nn.Linear(in_features=cfg['hidden_dim'],
                                       out_features=cfg['hidden_dim'])

    def forward(self, x):
        """
            x.shape = (batch_size, vis_size, hidden_dim)

        """

        x = rearrange(x, 'b l f -> b f l')  # shape: tch_size, hidden_dim, vis_size)
        x = self.sequence_inter(x)  # shape: (batch_size, hidden_dim, ts_size)

        x = rearrange(x, 'b f l -> b l f')  # shape: (batch_size, ts_size, hidden_dim)
        x = self.feature_inter(x)  # shape: (batch_size, ts_size, hidden_dim)

        return x
    

class StockEmbedding(nn.Module):
    def __init__(self, cfg: dict) -> None:

        """
        Args:
            cfg (dict):     {
                                'z_dim': int = 6,  # Number of Features
                                'ts_size': int = 24  # Time-Series size,
                                'mask_size': int = 1,
                                'num_masks': int = 3,
                                'hidden_dim': int = 12,
                                'embed_dim': int = 6,
                                'num_layers': int = 3,
                                'num_embed': int = 32,
                            }

        """
        
        super().__init__()

        self.config = cfg
        
        self.config['total_mask_size'] = self.config['num_masks'] * self.config['mask_size']
        
        self.encoder = Encoder(cfg=self.config)

        self.interpolator = Interpolator(cfg=self.config)

        self.decoder = Decoder(cfg=self.config)


        print('StockEmbedding initialized')


    def forward_ae(self, x: torch.Tensor):
        """
            mae_pseudo_mask is equivalent to the Autoencoder
            There is no interpolator in this mode

        Args:
            x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """
        
        x_enc = self.encoder(x)

        x_dec = self.decoder(x_enc)
        
        return x_enc, x_dec

    
    def forward_mae(self, x: torch.Tensor, masks: torch.Tensor):
        """
            No mask tokens, using Interpolation in the latent space

        Args:
            x (torch.Tensor):       --> shape: (batch_size, ts_size, z_dim)
            masks (torch.Tensor)    --> shape: (batch_size, ts_size), với mỗi giá trị là True hoặc False (True nghĩa là bị mask)

        """
        
        x_vis = mask_it(x, masks=masks)  # (batch_size, vis_size, z_dim)

        x_enc = self.encoder(x_vis)  # (batch_size, vis_size, hidden_dim)

        x_inter = self.interpolator(x_enc)  # (batch_size, ts_size, hidden_dim)

        x_dec = self.decoder(x_inter)  # (batch_size, ts_size, z_dim)


        return x_enc, x_inter, x_dec


class StockEmbedderLightning(L.LightningModule):
    def __init__(self, cfg: dict, is_training: bool = False, override_cfg: dict = None, program_logger = None):
        """
        Args:
            cfg (dict):     {
                                'training': {  # if is_training
                                    'mode': int = 3  # 1: train_ae | 2: train_embed | 3: train_recon
                                },
                                'model': {
                                    'z_dim': int = 6,  # Number of Features
                                    'ts_size': int = 24  # Time-Series size,
                                    'mask_size': int = 1,
                                    'num_masks': int = 3,
                                    'hidden_dim': int = 12,
                                    'embed_dim': int = 6,
                                    'num_layers': int = 3,
                                    'num_embed': int = 32,
                                }
                            }

        """

        super().__init__()

        if override_cfg:
            cfg.update(override_cfg)

        self.config = cfg

        self.is_training = is_training
        
        self.program_logger = program_logger

        if self.is_training:
            self.save_hyperparameters({'cfg': cfg})
        
        self.model = StockEmbedding(cfg=self.config['model'])

        self.criterion = nn.MSELoss(reduction='mean')


        print('StockEmbedderLightning initialized')

    
    def check_config(self):
        """
            *   Check config
        """

        if self.is_training:
            if self.config['training']['mode'] not in [1, 2, 3]:
                raise Exception('training_mode must be: 1 or 2 or 3.')


    def get_embedding(self, x: torch.Tensor):
        """
            defines the prediction/inference actions
            
            *   INPUT:
                        x (torch.Tensor):       --> shape: (batch_size, ts_size, z_dim)
                                                --> NORMALIZED using scaler
            *   OUTPUT:
                        stock_embedding -->  shape: (batch_size, ts_size, z_dim)

        """

        self.model.eval()  # Đảm bảo mô hình ở chế độ đánh giá
        with torch.no_grad():  # Tắt tính toán gradient để tăng tốc độ và tiết kiệm bộ nhớ
            x_enc, x_dec = self.model.forward_ae(x)
        
        return x_enc
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer
    

    def ae_step(self, x: torch.Tensor):
        """
            ae_step defines the step in train loop. (mode: train_ae)

            Args:

                x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)
                                    --> NORMALIZED using scaler

        """

        batch_size, ts_size, z_dim = x.shape

        x_enc, x_dec = self.model.forward_ae(x)

        # Calculate loss
        loss = self.criterion(x_dec, x)


        return loss
    

    def embed_step(self, x: torch.Tensor):
        """
            embed_step defines the step in train loop. (mode: train_embed)

            Args:

                x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)
                                    --> NORMALIZED using scaler

        """

        batch_size, ts_size, z_dim = x.shape

        random_masks = generate_random_masks(num_samples=batch_size, ts_size=ts_size, mask_size=self.config['model']['mask_size'], num_masks=self.config['model']['num_masks'])  # shape: (batch_size, ts_size)

        # Get the target x_ori_enc by Autoencoder
        self.model.eval()
        x_enc_ae, x_dec_ae = self.model.forward_ae(x)
        x_enc_ae = x_enc_ae.clone().detach()  # shape: (batch_size, ts_size, hidden_dim)

        self.model.train()

        x_enc_mae, x_inter_mae, x_dec_mae = self.model.forward_mae(x, masks=random_masks)

        # Only calculate loss for those being masked
        x_inter_mae_masked = x_inter_mae[random_masks].reshape(batch_size, -1, self.config['model']['hidden_dim'])
        x_enc_ae_masked = x_enc_ae[random_masks].reshape(batch_size, -1, self.config['model']['hidden_dim'])

        loss = self.criterion(x_inter_mae_masked, x_enc_ae_masked)

        # # By annotate lines above, we take loss on all patches
        # loss = self.criterion(x_inter_mae, x_enc_ae)  # embed_loss


        return loss
    

    def recon_step(self, x: torch.Tensor):
        """
            recon_step defines the step in train loop. (mode: train_recon)

            Args:

                x (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)
                                    --> NORMALIZED using scaler

        """

        batch_size, ts_size, z_dim = x.shape

        random_masks = generate_random_masks(num_samples=batch_size, ts_size=ts_size, mask_size=self.config['model']['mask_size'], num_masks=self.config['model']['num_masks'])  # shape: (batch_size, ts_size)

        x_enc, x_inter, x_dec = self.model.forward_mae(x, masks=random_masks)

        # Calculate loss
        loss = self.criterion(x_dec, x)


        return loss
    

    def training_step(self, batch, batch_idx):
        """
            training_step defines the train loop. It is independent of forward

            Args:
                batch (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """

        if self.config['training']['mode'] == 1:
            loss = self.ae_step(x=batch)
        
        elif self.config['training']['mode'] == 2:
            loss = self.embed_step(x=batch)
        
        else:
            loss = self.recon_step(x=batch)
        

        self.log(name='Loss/Train', value=loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)


        return loss
    

    def validation_step(self, batch, batch_idx):
        """
            Args:
                batch (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """

        if self.config['training']['mode'] == 1:
            loss = self.ae_step(x=batch)
        
        elif self.config['training']['mode'] == 2:
            loss = self.embed_step(x=batch)
        
        else:
            loss = self.recon_step(x=batch)
        

        self.log(name='Loss/Val', value=loss, prog_bar=True)


        return loss
    

    def test_step(self, batch, batch_idx):
        """
            Args:
                batch (torch.Tensor):   --> shape: (batch_size, ts_size, z_dim)

        """

        if self.config['training']['mode'] == 1:
            loss = self.ae_step(x=batch)
        
        elif self.config['training']['mode'] == 2:
            loss = self.embed_step(x=batch)
        
        else:
            loss = self.recon_step(x=batch)
        

        self.log(name='Loss/Test', value=loss, prog_bar=True)


        return loss
    
    
    def on_train_epoch_end(self):
        if self.program_logger:
            self.program_logger.info(f'Completed Training Epoch: {self.current_epoch}')