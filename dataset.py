# %%
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import io
import os
from tqdm import tqdm

# %%
# Replace 'data.csv' with the path to your CSV file
df_test = pd.read_csv("26a3f9eddc82d3907db7749c34f68938_test.csv")
df_train = pd.read_csv("26a3f9eddc82d3907db7749c34f68938_train.csv")
df_val = pd.read_csv("26a3f9eddc82d3907db7749c34f68938_val.csv")


# %%
# TEMPERATURE FOR TUNING OF PROBABILITY
T = 1.066


def sigmoid(x, T=T):
    return 1 / (1 + np.exp(-x / T))


# %%
sigmoid(0.4)

# %%
import os
import pandas as pd
import numpy as np
from PIL import Image


def process_row(row, save_dir="/mnt/nas05/data01/vincenzo/ecallisto/hu_dataset"):
    # Read the Parquet file
    parquet_path = row["file_path"]
    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return None

    df_parquet = pd.read_parquet(parquet_path)

    # Convert DataFrame to numpy array
    data_array = df_parquet.values.T.astype(np.uint8)

    assert data_array.max() <= 255
    assert data_array.min() >= 0

    # Create an image from the array
    image = Image.fromarray(data_array, mode="L")

    # Create file path to save the image
    img_filename = f"{row['antenna']}_{row['datetime']}.png"
    img_save_path = os.path.join(save_dir, img_filename)

    # Save the image to disk
    image.save(img_save_path, format="PNG")

    # Prepare the example dictionary
    example = {
        "image": img_save_path,
        "manual_label": int(row["label"]),
        "logits": float(row["pred"]),
        "prob": sigmoid(row["pred"]),
        "model_label": int(row["pred"] > 0),
        "start_datetime": pd.to_datetime(row["datetime"]),
        "antenna": row["antenna"],
        "freq_axis": df_parquet.columns.values.astype(np.float64).round(2),
    }

    return example, df_parquet


# %%
examples_test = []
for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
    example, spectro = process_row(row)
    if example is not None:
        examples_test.append(example)


# %%
examples_train = []
for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
    example, spectro = process_row(row)
    if example is not None:
        examples_train.append(example)


# %%
examples_val = []
for index, row in tqdm(df_val.iterrows(), total=len(df_val)):
    example, spectro = process_row(row)
    if example is not None:
        examples_val.append(example)


# %%
from datasets import Dataset, DatasetDict
import datasets

dd = DatasetDict()
dd["test"] = Dataset.from_pandas(pd.DataFrame(examples_test)).cast_column(
    "image", datasets.Image(decode=False)
)
dd["val"] = Dataset.from_pandas(pd.DataFrame(examples_val)).cast_column(
    "image", datasets.Image(decode=False)
)
dd["train"] = Dataset.from_pandas(pd.DataFrame(examples_train)).cast_column(
    "image", datasets.Image(decode=False)
)

# %%
dd.push_to_hub(
    "i4ds/ecallisto_radio_sunburst", token="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"
)

# %%
