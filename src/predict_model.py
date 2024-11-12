import json
import argparse
import torch
import os
import sys
sys.path.append(os.getcwd())
from src.utils.logger_config import get_logger
from src.models.stock_predictor import StockPredictorLightning
import pandas as pd
import numpy as np


def load_config(config_file: str):
    with open(file=config_file, mode='r', encoding='UTF-8') as file:
        cfg = json.load(file)
    
    return cfg


def main(cfg: dict, stock_df: pd.DataFrame, verbose: bool = True, logger = None):
    """
    Args:
        stock_df (pd.DataFrame):
        ({
            'Date': pd.date_range(start='2024-04-30', periods=100),
            'Open': np.random.uniform(100, 300, size=100),
            'High': np.random.uniform(100, 300, size=100),
            'Low': np.random.uniform(100, 300, size=100),
            'Close': np.random.uniform(100, 300, size=100),
            'Adj_Close': np.random.uniform(100, 300, size=100),
            'Volume': np.random.uniform(100, 300000, size=100)
            # **{f: np.random.uniform(100, 300, size=100) for f in ['stoch', 'adx', 'bollinger_hband', 'mfi', 'rsi', 'ma', 'std', 'adl', 'williams', 'macd', 'obv', 'sar', 'ichimoku_a', 'ichimoku_b']}
        })
        
    """

    if verbose:
        if not logger:
            raise Exception('logger must be provided if verbose is True.')
    
    
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.sort_values(by='Date', inplace=True)
    stock_df.reset_index(drop=True, inplace=True)
    
    stock_features = stock_df.drop(columns=['Date']).columns.to_list()
    
    
    # Load Model
    stock_predictor = StockPredictorLightning.load_from_checkpoint(
        checkpoint_path=cfg['checkpoint_path'],
        strict=False,
        is_training = False,
        override_cfg = {
            'data': {
                'stock_features': stock_features
            }
        }
    )
    
    # Bạn có thể tùy chọn chuyển mô hình sang chế độ eval
    stock_predictor.eval()

    if verbose:
        logger.info(f"stock_predictor.config = {stock_predictor.config}")
        logger.info(f'stock_predictor.is_training = {stock_predictor.is_training}')
    
    # Dự đoán trên dữ liệu mới
    input_stock = stock_df.iloc[ - stock_predictor.stock_embedder.config['model']['ts_size'] : ][stock_features].values
    input_stock = torch.tensor(input_stock.astype(np.float32), dtype=torch.float32)  # shape: (ts_size, features)
    
    # Add batch dimension
    input_stock = input_stock.unsqueeze(dim=0)  # shape: (1, ts_size, features)
    
    with torch.no_grad():
        prediction = stock_predictor.predict(stock=input_stock, future_days=cfg['future_days'])
    
    
    predicted_stock = prediction.values  # shape: (batch_size, future_days, stock_features)
    predicted_stock = predicted_stock[0]  # shape: (future_days, stock_features)
    predicted_stock = predicted_stock.cpu().numpy()
    
    
    prediction_df = pd.DataFrame(
        data=predicted_stock,
        columns=prediction.feature_names_in_
    )
    
    
    return prediction_df


if __name__ == '__main__':
    # --------------- Data ----------------
    stock_df = pd.DataFrame({
        'Date': pd.date_range(start='2024-04-30', periods=100),
        'Open': np.random.uniform(100, 300, size=100),
        'High': np.random.uniform(100, 300, size=100),
        'Low': np.random.uniform(100, 300, size=100),
        'Close': np.random.uniform(100, 300, size=100),
        'Adj_Close': np.random.uniform(100, 300, size=100),
        'Volume': np.random.uniform(100, 300000, size=100)
        # **{f: np.random.uniform(100, 300, size=100) for f in ['stoch', 'adx', 'bollinger_hband', 'mfi', 'rsi', 'ma', 'std', 'adl', 'williams', 'macd', 'obv', 'sar', 'ichimoku_a', 'ichimoku_b']}
    })
    
    # --------------- END: Data ----------------
    
    
    # Tạo parser và thêm tham số cho file cấu hình
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, help='Path to config file (JSON format).',
                        default='config/predict_config.json')

    args = parser.parse_args()

    # Tải cấu hình từ file được chỉ định
    cfg = load_config(config_file=args.config_file)
    

    # Run
    
    logger = get_logger(name=__name__, log_file=os.path.join(cfg['output_dir'], 'logs.log'), mode='w')
    
    
    prediction_df = main(cfg=cfg, stock_df=stock_df, verbose=cfg['verbose'], logger=logger)
    
    
    prediction_df.to_csv(path_or_buf=os.path.join(cfg['output_dir'], 'prediction.csv'), encoding='UTF-8', index=False)
    
    logger.info(f'Saved Prediction to {cfg["output_dir"]}')
    
    
    logger.info('Finished.')