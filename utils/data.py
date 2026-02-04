import pandas as pd


def load_data(csv_path: str, require_disease: bool = False) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "iso-8859-1"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc, quotechar='"')
            break
        except Exception as e:
            last_err = e

    if df is None:
        raise RuntimeError(f"Impossible de lire le csv {csv_path}: {last_err}")

    df.columns = df.columns.str.strip()

    if "description" not in df.columns:
        if len(df.columns) >= 1:
            df = df.rename(columns={df.columns[0]: "description"})
        else:
            raise ValueError("Colonne description manquante")

    if require_disease and "disease" not in df.columns:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]: "disease"})
        else:
            raise ValueError("Colonne disease manquante")

    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].notna() & (df["description"] != "")]

    if "disease" in df.columns:
        df["disease"] = df["disease"].astype(str).str.strip()

    return df

