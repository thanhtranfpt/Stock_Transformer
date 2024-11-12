import json
import argparse
import lightning as L
# import pytorch_lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
# from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
import os
import sys
sys.path.append(os.getcwd())
from src.utils.logger_config import get_logger
from src.models.stock_predictor import StockPredictorLightning
from src.models.stock_embedder import StockEmbedderLightning
from src.data.make_dataset import create_dataset, create_dataloaders
# from lightning.pytorch.strategies import FSDPStrategy
import joblib


def load_config(config_file: str):
    with open(file=config_file, mode='r', encoding='UTF-8') as file:
        cfg = json.load(file)
    
    return cfg


def main(cfg: dict, verbose: bool = True, logger = None):
    if verbose:
        if not logger:
            raise Exception('logger must be provided if verbose is True.')
        

    early_stopping = EarlyStopping(
        monitor='Loss/Val',   # Theo dõi 'val_loss'
        patience=3,           # Số lượng epoch không cải thiện trước khi dừng
        mode='min'            # 'min' để dừng nếu 'val_loss' không giảm (vì loss càng nhỏ càng tốt)
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='Loss/Val',  # Lưu checkpoint tốt nhất dựa trên 'val_loss'
        save_top_k=1,        # Chỉ lưu checkpoint tốt nhất (top 1)
        save_last=True,      # Luôn lưu model cuối cùng
        mode='min',          # Dựa trên giảm 'val_loss'
        filename='{epoch}-{step}-best-checkpoint',  # Tên tệp checkpoint
        verbose=True               # In thông báo khi lưu checkpoint
    )

    # Khởi tạo TensorBoard Logger
    # tensorboard_logger = TensorBoardLogger(
    #     save_dir=os.path.join(
    #         cfg['output_dir'],
    #         'lightning_logs',
    #         'tensorboard'
    #     ),
    #     name=cfg['model_name']
    # )

    wandb_logger = WandbLogger(
        name=cfg['model_name'],
        save_dir=os.path.join(
            cfg['output_dir'],
            'lightning_logs',
            'wandb'
        ),
        version=cfg['version'],
        project=cfg['model_name'],
        log_model='all'
    )

    # Tạo Callback để Log GPU Usage
    # gpu_logger = GPUUsageLogger(log_interval=10)  # Log GPU usage sau mỗi 10 batches

    # trainer = pl.Trainer(
    #     logger=tensorboard_logger,
    #     min_epochs=10, max_epochs=200, max_steps=5000,
    #     callbacks=[
    #         gpu_logger,
    #         checkpoint_callback
    #         # early_stopping
    #     ],
    #     precision=16,
    #     accelerator = 'gpu',
    #     devices=[0]
    # )

    trainer = L.Trainer(
        logger=wandb_logger,
        min_epochs=cfg['trainer']['min_epochs'],
        max_epochs=cfg['trainer']['max_epochs'],
        max_steps=cfg['trainer']['max_steps'],
        callbacks=[
            checkpoint_callback
            # early_stopping
        ],
        precision=16,
        accelerator='cuda',
        devices=[0],  # [0, 1] or 'auto'
        strategy = 'auto'  # FSDPStrategy()
    )
    
    
    # Create dataset and dataloaders
    logger.info('Creating Dataset and Dataloaders')
    
    dataset = create_dataset(
        cfg={
            **cfg['dataset'],
            'ts_size': StockEmbedderLightning.load_from_checkpoint(checkpoint_path=cfg['stock_predictor_lightning']['stock_embedder']['checkpoint_path'], is_training = False, strict=False).config['model']['ts_size']
        },
        logger=logger
    )
    
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset=dataset, cfg=cfg['dataloaders'])
    
    if verbose:
        logger.info('Dataset and Dataloaders created successfully.')
        
        
    # Initialize Model
    
    cfg['stock_predictor_lightning']['data'] = {
        **cfg['stock_predictor_lightning']['data'],
        'stock_features': dataset.stock_features
    }
    
    if cfg['training']['resume']:
        model = StockPredictorLightning.load_from_checkpoint(
            checkpoint_path=cfg['training']['checkpoint_path'],
            is_training = True,
            override_cfg = cfg['stock_predictor_lightning'],
            program_logger = logger
        )
    
    else:
        model = StockPredictorLightning(
            cfg=cfg['stock_predictor_lightning'],
            is_training=True,
            override_cfg=cfg['stock_predictor_lightning'],
            program_logger=logger
        )
        
        
    if verbose:
        logger.info('StockPredictorLightning created successfully')
    
    
    # Run: Training
    
    if verbose:
        logger.info('Start Training')
        
    trainer.fit(
        model=model,
        ckpt_path = cfg['training']['checkpoint_path'],
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    if verbose:
        logger.info(f'Completed Training.')
        

    # Run: Testing
    
    trainer.test(
        model=model,
        ckpt_path='best',
        dataloaders=test_dataloader,
        verbose=True
    )

    if verbose:
        logger.info(f'Done Testing.')
    
    # Save
    joblib.dump(cfg, filename=os.path.join(cfg['output_dir'], 'config.pkl'))

    if verbose:
        logger.info('Saved config.')


if __name__ == '__main__':
    # Tạo parser và thêm tham số cho file cấu hình
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, help='Path to config file (JSON format).',
                        default='config/train_config.json')

    args = parser.parse_args()

    # Tải cấu hình từ file được chỉ định
    cfg = load_config(config_file=args.config_file)

    # Main Processing
    logger = get_logger(
        name=__name__,
        log_file=os.path.join(
            cfg['output_dir'],
            'logs.log'
        ),
        mode='w'
    )

    logger.info(f'Start Main Processing:')


    main(cfg=cfg, verbose=cfg['verbose'], logger=logger)


    logger.info(f'Finished!')