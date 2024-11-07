# Stock_Transformer

Stock Predictor (No News) of The Finance Forecasting Project


# Guide to Training

**Step 1**. Install: `pip install -r requirements.txt`

**Step 2**. Config in the file: `config/train_config.json`

    ```json
    {
        "model_name": str = "Stock_Predictor_No_News_Lightning",
        "version": str,
        "training": {
            "resume": bool = false,
            "checkpoint_path": null  # if not resume. Else, str
        },
        "trainer": {
            "min_epochs": int = 1,
            "max_epochs": int = 5,
            "max_steps": int = -1
        },
        "dataset": {
            "stock_file": str
        },
        "dataloaders": {
            "split_ratio": {
                "train": float = 0.8,
                "val": float = 0.1,
                "test": float = 0.1
            },
            "batch_size": int = 32
        },
        "stock_predictor_lightning": {
            "data": {},
            "stock_embedder": {  # if not resume. Else, [Optional]
                "checkpoint_path": str,
                "scaler_file": str
            },
            "transformer": {  # if not resume. Else, [Removed]
                "pos_encoder": {
                    "d_model": int = 10,
                    "max_len": int = 5000
                },
                "num_encoders": int = 5,
                "num_decoders": int = 5,
                "encoder": {
                    "d_model": int = 10,
                    "attention_heads": int = 4,
                    "feedforward": {
                        "d_model": int = 10,
                        "d_ff": int = 2048,
                        "dropout": float = 0.1
                    }
                },
                "decoder": {
                    "d_model": int = 10,
                    "attention_heads": int = 4,
                    "feedforward": {
                        "d_model": int = 10,
                        "d_ff": int = 2048,
                        "dropout": float = 0.1
                    }
                },
                "final_linear": {
                    "in_features": int = 10,
                    "out_features": int = 6
                }
            }
        },
        "output_dir": str,
        "verbose": bool = true
    }

**Step 3**. Run in terminal: `python src/train_model.py`


# Guide to Inference

**Step 1**. Install: `pip install -r requirements.txt`

**Step 2**. Config in the file: `config/predict_config.json`

    ```json
    {
        "checkpoint_path": str,
        "future_days": int = 7,
        "output_dir": str,
        "verbose": bool = true
    }

**Step 3**. Run in terminal: `python src/predict_model.py`


# Models Trained

- **Stock Prediction Models**:
    - **Models_Notes**: https://docs.google.com/spreadsheets/d/1MqzhONnLtlk43WdhiDtaG2MItoXea6-KWEeAKTa2cPM/edit?usp=sharing

- **Stock Embedding Models**:
    - **ver_1**:
        - **Link**: https://drive.google.com/drive/folders/1wL1GAkzCax71vM_7tXYlpiSFcGNVb20F?usp=drive_link
        - Author's Pretrained
    - **Models_Notes**: https://docs.google.com/spreadsheets/d/1thZECwq45yi7cjM8rINQkrbYbnbAtw6wWsCQIW1g118/edit?usp=drive_link


# External Stock Data

- **ver_1**
    - **Link**: https://drive.google.com/drive/folders/1MNPU6IWEJxJgCCwGRgy-DsIvFI5SP7K8?usp=drive_link
    - **Source**: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
    - **Date Collected (Download Date)**: 2nd October 2024
    - **Data Time Range**: From January 4, 2010 to October 2, 2020
    - **Symbols Included**: All S&P 500 companies (503 symbols)
    - **Columns**:
        - **sp500_stocks.csv**: `['Date', 'Symbol', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']`
    - **Comments**:


# Processed Stock Data

- **ver_1**
    - **Link**: https://drive.google.com/drive/folders/1U7CkpnMOG0GKFCOH98a92r9l1p7zdtu2?usp=drive_link
    - **Source**: Derived from https://drive.google.com/drive/folders/1MNPU6IWEJxJgCCwGRgy-DsIvFI5SP7K8?usp=drive_link
    - **Symbols Included**: All S&P 500 companies (503 symbols)
    - **Data Time Range**: From January 4, 2010 to October 2, 2020
    - **Columns**: `['Date', 'Symbol', 'Adj_Close', 'Close', 'High', 'Low', 'Open', 'Volume']`
    - **Comments**: