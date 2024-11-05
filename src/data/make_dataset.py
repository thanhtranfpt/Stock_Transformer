from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


class StockNewsDataset(Dataset):
    def __init__(self, samples: list, stock_features: list):
        
        self.samples = samples
        self.stock_features = stock_features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        sample = self.samples[idx]
        
        return {
            'input': {
                'stock': torch.tensor(sample['input_stock_data'].astype(np.float32), dtype=torch.float32)
            },
            'target': {
                'stock': torch.tensor(sample['target_stock_data'].astype(np.float32), dtype=torch.float32)
            }
        }


def create_stock_data(stock_df: pd.DataFrame, ts_size: int, logger):
  
    # Check
    if not all(col in stock_df.columns for col in ['Date', 'Symbol']):
        logger.error(f"One or both of 'Date' and 'Symbol' are missing from stock_df.")
        
        raise Exception(f"One or both of 'Date' and 'Symbol' are missing from stock_df.")
    
    # Create data
    samples = []
    symbols = stock_df['Symbol'].unique()
    stock_features = stock_df.drop(columns=['Date', 'Symbol']).columns.tolist()
    
    for idx, symbol in enumerate(tqdm(symbols)):
        # Lọc dữ liệu cho mỗi symbol
        symbol_df = stock_df[stock_df['Symbol'] == symbol].sort_values(by='Date').reset_index(drop=True)

        for i in range(len(symbol_df) - ts_size):
            # Get stock data
            input_stock_data = symbol_df.iloc[i : i + ts_size][stock_features].values
            target_stock_data = symbol_df.iloc[i + ts_size - 1][stock_features].values
            
            samples.append(
                {
                    'input_stock_data': input_stock_data,
                    'target_stock_data': target_stock_data
                }
            )
            
        
        logger.info(f'Completed for Symbol: {symbol} ({idx + 1} / {len(symbols)}. Number of Samples so far: {len(samples)}')
                
    
    return {
        'stock_features': stock_features,
        'samples': samples
    }


def create_dataset(cfg: dict, logger):
    """

    Args:
        cfg (dict): {
            'stock_file': str,
            'ts_size': int = 24
        }
        
        logger (_type_): _description_. Defaults to None.
        
    """
    
    stock_df = pd.read_csv(cfg['stock_file'], encoding='UTF-8')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Create Dataset
    logger.info('Start Creating Stock Data ...')
    
    stock_news_data = create_stock_data(stock_df=stock_df, ts_size=cfg['ts_size'], logger=logger)
    
    logger.info('Completed creating Stock Data')
    
    dataset = StockNewsDataset(samples=stock_news_data['samples'],
                               stock_features=stock_news_data['stock_features'])
    
    return dataset


def create_dataloaders(dataset: Dataset, cfg: dict):
    """
    Args:
        cfg (dict):     {
                            'split_ratio': {
                                'train': 0.8,  # 80% for training
                                'val': 0.1,  # 10% for validation
                                'test': 0.1  # 10% for testing
                            },
                            'batch_size': 32
                        }
    """

    # Random split
    train_size = int(cfg['split_ratio']['train'] * len(dataset))
    val_size = int(cfg['split_ratio']['val'] * len(dataset))
    test_size = len(dataset) - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, val_size, test_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    
    return train_dataloader, val_dataloader, test_dataloader