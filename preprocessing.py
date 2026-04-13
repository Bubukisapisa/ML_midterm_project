import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import TargetEncoder


df_original = pd.read_csv("bank-additional-full.csv", sep=";")
df = df_original.copy()

# ─────────────────────────────────────────────────────────────────
# БЛОК 1 — Трансформації ДО розділення даних
# ─────────────────────────────────────────────────────────────────

def preproc_bef_split(df_orig):
    df = df_orig.copy()
    # Видалення колонок
    df = df.drop(columns=["duration", "housing", "loan", "day_of_week"])

    # Видалення аномальних записів
    df = df[df["campaign"] != 56]

    # pdays → was_contacted + pdays_real, видалити оригінальну
    df["was_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays_real"]    = df["pdays"].where(df["pdays"] != 999, other=0)
    df = df.drop(columns=["pdays"])

    # default: злити "yes" + "unknown" → "not_confirmed_no"
    df["default"] = df["default"].replace({"yes": "not_confirmed_no",
                                            "unknown": "not_confirmed_no"})
    df["default"] = (df["default"] == 'no').astype(int)

    # education: злити "illiterate" → "unknown"
    df["education"] = df["education"].replace({"illiterate": "unknown"})

    # contact → бінарна: cellular=1, telephone=0
    df["contact"] = (df["contact"] == "cellular").astype(int)

    # Таргет: yes→1, no→0
    df["y"] = (df["y"] == "yes").astype(int)

    return df


# ─────────────────────────────────────────────────────────────────
# БЛОК 2 — Розділення даних (60% train / 20% val / 20% test)
# ─────────────────────────────────────────────────────────────────
def split_data(df_prep):
    df = df_prep.copy()

    y = df["y"]
    X = df.drop(columns=["y"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25, random_state=42, stratify=y_train_full
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────────────────────────
# БЛОК 3 — Трансформації ПІСЛЯ розділення (фіт тільки на train)
# ─────────────────────────────────────────────────────────────────

def transform_after_split(X_train, X_val, X_test, y_train):
    
    # ── Log-трансформація ────────────────────────────────────────────
    log_cols = ["campaign", "pdays_real"]

    for col in log_cols:
        X_train[col] = np.log1p(X_train[col])
        X_val[col]   = np.log1p(X_val[col])
        X_test[col]  = np.log1p(X_test[col])

    # ── StandardScaler ───────────────────────────────────────────────
    num_cols = ["age", "campaign", "pdays_real", "previous",
                "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                "euribor3m", "nr.employed"]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # ── Ordinal Encoding — education ─────────────────────────────────
    edu_order = [["basic.4y", "basic.6y", "basic.9y",
                   "high.school", "professional.course", "university.degree"]]

    ord_enc = OrdinalEncoder(categories=edu_order,
                             handle_unknown="use_encoded_value", unknown_value=-1)
    X_train["education"] = ord_enc.fit_transform(X_train[["education"]])
    X_val["education"]   = ord_enc.transform(X_val[["education"]])
    X_test["education"]  = ord_enc.transform(X_test[["education"]])

    # ── OHE — marital, poutcome ──────────────────────────────────────
    ohe_cols = ["marital", "poutcome"]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(X_train[ohe_cols])

    def ohe_transform(X, ohe, ohe_cols):
      transformed = ohe.transform(X[ohe_cols])
      new_cols = ohe.get_feature_names_out(ohe_cols)
      transformed_df = pd.DataFrame(transformed, columns=new_cols, index=X.index)
      return pd.concat([X.drop(columns=ohe_cols), transformed_df], axis=1)

    X_train = ohe_transform(X_train, ohe, ohe_cols)
    X_val   = ohe_transform(X_val,   ohe, ohe_cols)
    X_test  = ohe_transform(X_test,  ohe, ohe_cols)

    # ── Target Encoding — job, month ─────────────────────────────────
    
    te = TargetEncoder(smooth="auto")

    X_train[["job", "month"]] = te.fit_transform(X_train[["job", "month"]], y_train)
    X_val[["job", "month"]]   = te.transform(X_val[["job", "month"]])
    X_test[["job", "month"]]  = te.transform(X_test[["job", "month"]])


    print("Готово!")
    print(f"X_train shape: {X_train.shape}")
    print(f"Колонки: {list(X_train.columns)}")

    return X_train, X_val, X_test
