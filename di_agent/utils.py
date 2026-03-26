from __future__ import annotations

import pandas as pd


def save_uploaded_file(uploaded_file) -> str:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_dataframe_info(df: pd.DataFrame) -> str:
    buf: list[str] = []
    buf.append(f"1. 列名列表: {list(df.columns)}")
    buf.append(f"2. 数据类型:\n{df.dtypes.to_string()}")
    buf.append(f"3. 前3行数据预览:\n{df.head(3).to_string()}")
    return "\n".join(buf)

