import lightning as L
# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.stock_embedder import StockEmbedderLightning
from src.models.transformer import Transformer
import joblib


class StockPredictionOutput:
    def __init__(self, values: torch.Tensor, feature_names_in_: list) -> None:
        self.values = values
        self.feature_names_in_ = feature_names_in_


class StockPredictorLightning(L.LightningModule):
    def __init__(self, cfg: dict, is_training: bool = False, override_cfg: dict = None, program_logger = None) -> None:
        """

        Args:
            cfg (dict): {
                'data': {
                    'stock_features': list = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
                },
                'stock_embedder': {
                    'checkpoint_path': str,
                    'scaler_file': str,
                },
                'transformer': {
                    'pos_encoder': {
                        'd_model': 10,
                        'max_len': 5000
                    },
                    'num_encoders': 5,
                    'num_decoders': 5,
                    'encoder': {
                        'd_model': 10,
                        'attention_heads': 4,
                        'feedforward': {
                            'd_model': 10,
                            'd_ff': 2048,
                            'dropout': 0.1
                        }
                    },
                    'decoder': {
                        'd_model': 10,
                        'attention_heads': 4,
                        'feedforward': {
                            'd_model': 10,
                            'd_ff': 2048,
                            'dropout': 0.1
                        }
                    },
                    'final_linear': {
                        'in_features': 10,
                        'out_features': 10
                    }
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
        
        # Load Stock Embedder
        self.stock_embedder = StockEmbedderLightning.load_from_checkpoint(
            checkpoint_path=self.config['stock_embedder']['checkpoint_path'],
            is_training = False,
            strict=False
        )
        self.stock_embedder.eval()
        
        # Đặt requires_grad = False cho tất cả các tham số
        for param in self.stock_embedder.parameters():
            param.requires_grad = False
        
        self.stock_scaler = joblib.load(filename=self.config['stock_embedder']['scaler_file'])
        self.stock_scaler_features_mapping = [self.config['data']['stock_features'].index(feature) for feature in self.stock_scaler.feature_names_in_]
        
        self.transformer = Transformer(cfg=self.config['transformer'])
        
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.check_config()
        
        
        print('StockPredictor initialized.')
    
    
    def check_config(self):
        if self.config['transformer']['final_linear']['out_features'] != len(self.stock_scaler.feature_names_in_):
            raise ValueError(f"out_features of final_linear of transformer ({self.config['transformer']['final_linear']['out_features']}) must be equal to feature_names_in_ of stock_scaler ({len(self.stock_scaler.feature_names_in_)}: {self.stock_scaler.feature_names_in_})")
    
    
    def normalize_stock_data(self, stock_data: torch.Tensor):
        """
            *   INPUT: stock_data   --> shape: (batch_size, ts_size, cfg['data']['stock_features'])
                                    --> original - NOT normalized
            *   OUTPUT: stock_data      --> shape: (batch_size, ts_size, cfg['data']['stock_features'])
                                        --> NORMALIZED
        """
        
        batch_size, ts_size, n_features = stock_data.shape
        
        # Normalize
        stock_data = stock_data.reshape(-1, n_features)  # Shape: (batch_size * ts_size, features)
        stock_data = stock_data.cpu().numpy()
        stock_data = stock_data[ : , self.stock_scaler_features_mapping]
        stock_data = self.stock_scaler.transform(stock_data)
        stock_data = torch.tensor(stock_data, dtype=torch.float32)
        stock_data = stock_data.view(batch_size, ts_size, n_features)  # shape: (batch_size, ts_size, input_stock_features)
        stock_data = stock_data.to(self.device)
        
        return stock_data
    
        
    def get_stock_embedding(self, stock_data: torch.Tensor):
        """
            *   INPUT: stock_data   --> shape: (batch_size, ts_size, stock_features)
                                    --> NORMALIZED using stock scaler
            *   OUTPUT: stock_embedding   --> shape: (batch_size, stock_dim)
        """
        
        with torch.no_grad():
            stock_embedding = self.stock_embedder.get_embedding(stock_data)  # shape: (batch_size, ts_size, stock_dim)
        
        return stock_embedding
    
    
    def predict(self, stock: torch.Tensor, future_days: int = 7):
        """
            defines the prediction/inference actions
        
        Args:
            stock: torch.Tensor  # shape: (batch_size, ts_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
            
        """
        
        # For Encoder
        stock = self.normalize_stock_data(stock_data=stock)  # shape: (batch_size, ts_size, stock_features)
        
        stock_embedding = self.get_stock_embedding(stock_data=stock)  # shape: (batch_size, ts_size, stock_dim)
        
        # For Decoder
        stock_last_days = stock[:, -1:, :]  # shape: (batch_size, 1, stock_features)
        
        for i in range(future_days):
            stock_last_days_embedding = self.get_stock_embedding(stock_data=stock_last_days)  # shape: (batch_size, n_stock_days, stock_dim)
            
            self.transformer.eval()
            with torch.no_grad():
                predicted_stock = self.transformer.forward(input_embedding=stock_embedding, output_embedding=stock_last_days_embedding)  # shape: (batch_size, stock_features)
                
            stock_last_days = torch.stack(tensors=[stock_last_days, predicted_stock.unsqueeze(dim=1)], dim=1)  # shape: (batch_size, n_stock_days, stock_features)
            
            if i == 0:
                stock_last_days = stock_last_days[:, 1:, :]  # shape: (batch_size, n_stock_days, stock_features)
        
        # Re-normalized stock predicted
        batch_size, n_stock_days, n_stock_features = stock_last_days.shape
        
        stock_last_days = stock_last_days.reshape(-1, n_stock_features)  # Shape: (batch_size * n_stock_days, stock_features)
        stock_last_days = stock_last_days.cpu().numpy()
        stock_last_days = self.stock_scaler.inverse_transform(stock_last_days)
        stock_last_days = torch.tensor(stock_last_days, dtype=torch.float32)
        stock_last_days = stock_last_days.view(batch_size, n_stock_days, n_stock_features)  # shape: (batch_size, n_stock_days, stock_features)
        
        
        return StockPredictionOutput(
            values=stock_last_days,
            feature_names_in_=self.stock_scaler.feature_names_in_
        )
    
    
    def forward(self, stock: torch.Tensor):
        """defines the forward in training

        Args:
            stock: torch.Tensor  # shape: (batch_size, ts_size, stock_features); NORMALIZED using stock scaler
        
        Returns:
            stock_features: torch.Tensor  # shape: (batch_size, stock_features); NORMALIZED
            
        """
        
        # For Encoder
        stock_embedding = self.get_stock_embedding(stock_data=stock)  # shape: (batch_size, ts_size, stock_dim)
        
        # For Decoder
        stock_last_day_embedding = self.get_stock_embedding(stock_data=stock[:, -1:, :])  # shape: (batch_size, 1, stock_dim)
        
        predicted_stock = self.transformer.forward(input_embedding=stock_embedding, output_embedding=stock_last_day_embedding)  # shape: (batch_size, stock_features)
        
        return predicted_stock
    
    
    def training_step(self, batch: dict):
        """training_step defines the train loop. It is independent of forward

        Args:
            batch (dict): {
                'input': {
                    'stock': torch.Tensor  # shape: (batch_size, ts_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
                },
                'target': {
                    'stock': torch.Tensor  # shape: (batch_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
                }
            }
        
        Returns:
            loss
            
        """
        
        input_stock = batch['input']['stock']  # shape: (batch_size, ts_size, cfg['data']['stock_features'])
        target_stock = batch['target']['stock']  # shape: (batch_size, cfg['data']['stock_features'])
        
        input_stock = self.normalize_stock_data(stock_data=input_stock)  # shape: (batch_size, ts_size, stock_features)
        
        predicted_stock = self.forward(stock=input_stock)  # shape: (batch_size, stock_features)
        
        # Calculate Loss
        loss = self.criterion(predicted_stock, target_stock[:, self.stock_scaler_features_mapping])
        
        self.log(name="Loss/Train", value=loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        
        return loss
    
    
    def validation_step(self, batch: dict):
        """
        
        Args:
            batch (dict): {
                'input': {
                    'stock': torch.Tensor  # shape: (batch_size, ts_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
                },
                'target': {
                    'stock': torch.Tensor  # shape: (batch_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
                }
            }
        
        Returns:
            loss
            
        """
        
        input_stock = batch['input']['stock']  # shape: (batch_size, ts_size, cfg['data']['stock_features'])
        target_stock = batch['target']['stock']  # shape: (batch_size, cfg['data']['stock_features'])
        
        input_stock = self.normalize_stock_data(stock_data=input_stock)  # shape: (batch_size, ts_size, stock_features)
        
        predicted_stock = self.forward(stock=input_stock)  # shape: (batch_size, stock_features)
        
        # Calculate Loss
        loss = self.criterion(predicted_stock, target_stock[:, self.stock_scaler_features_mapping])
        
        self.log(name="Loss/Val", value=loss, prog_bar=True)
        
        return loss
    
    
    
    def test_step(self, batch: dict):
        """
        
        Args:
            batch (dict): {
                'input': {
                    'stock': torch.Tensor  # shape: (batch_size, ts_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
                },
                'target': {
                    'stock': torch.Tensor  # shape: (batch_size, cfg['data']['stock_features']); original stock data - NOT normalized yet
                }
            }
        
        Returns:
            loss
            
        """
        
        input_stock = batch['input']['stock']  # shape: (batch_size, ts_size, cfg['data']['stock_features'])
        target_stock = batch['target']['stock']  # shape: (batch_size, cfg['data']['stock_features'])
        
        input_stock = self.normalize_stock_data(stock_data=input_stock)  # shape: (batch_size, ts_size, stock_features)
        
        predicted_stock = self.forward(stock=input_stock)  # shape: (batch_size, stock_features)
        
        # Calculate Loss
        loss = self.criterion(predicted_stock, target_stock[:, self.stock_scaler_features_mapping])
        
        self.log(name="Loss/Test", value=loss, prog_bar=True)
        
        
        return loss
    
    
    def on_train_epoch_end(self):
        if self.program_logger:
            self.program_logger.info(f'Completed Training Epoch: {self.current_epoch}')
    
    
    def configure_optimizers(self):
        
        # optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer
    

