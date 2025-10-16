import re
import pandas as pd
from pathlib import Path

def parse_neuralhydrology_log(log_path: str | Path):
    """Parse output.log from NeuralHydrology training run."""
    log_path = Path(log_path)
    text = log_path.read_text(encoding="utf-8")

    # Regex patterns
    train_pattern = r"Epoch (\d+) average loss: avg_loss: ([\d\.]+), avg_total_loss: ([\d\.]+)"
    val_pattern = (
        r"Epoch (\d+) average validation loss: ([\d\.]+).*?MAPE: ([\d\.]+), NSE: ([\d\.]+), "
        r"MSE: ([\d\.]+), RMSE: ([\d\.]+)"
    )

    # Extract
    train_data = re.findall(train_pattern, text)
    val_data = re.findall(val_pattern, text)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data, columns=["epoch", "avg_loss", "avg_total_loss"]).astype(float)
    val_df = pd.DataFrame(val_data, columns=["epoch", "avg_val_loss", "MAPE", "NSE", "MSE", "RMSE"]).astype(float)

    print(f"✅ Parsed {len(train_df)} training epochs and {len(val_df)} validation evaluations.")
    return train_df, val_df


