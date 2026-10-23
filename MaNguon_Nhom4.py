import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2, f as f_dist
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests
import pingouin as pg

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
np.set_printoptions(precision=4, suppress=True)

import urllib.request
import io

def nhanes_download(file_name):
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{file_name}.xpt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req).read()
    return pd.read_sas(io.BytesIO(raw), format="xport")

# Tải và đọc 6 module
glu    = nhanes_download("GLU_J")[["SEQN", "LBXGLU"]]
ghb    = nhanes_download("GHB_J")[["SEQN", "LBXGH"]]
hdl    = nhanes_download("HDL_J")[["SEQN", "LBDHDD"]]
trigly = nhanes_download("TRIGLY_J")[["SEQN", "LBXTR", "LBDLDL"]]
bpx    = nhanes_download("BPX_J")[["SEQN", "BPXSY1"]]
diq    = nhanes_download("DIQ_J")[["SEQN", "DIQ010"]]

# Merge 6 module theo cột khóa SEQN
df_raw = (glu
          .merge(ghb,    on="SEQN")
          .merge(hdl,    on="SEQN")
          .merge(trigly, on="SEQN")
          .merge(bpx,    on="SEQN")
          .merge(diq,    on="SEQN"))

print("Kích thước dữ liệu ban đầu:", df_raw.shape)
print(df_raw.head())


# 3.1. Tiền xử lý dữ liệu
# -----------------------
df_raw = df_raw[df_raw["DIQ010"].isin([1, 2, 3])]

bien_chinh = ["LBXGLU", "LBXGH", "LBXTR", "LBDHDD", "LBDLDL", "BPXSY1"]
df_clean = df_raw.dropna(subset=bien_chinh + ["DIQ010"]).reset_index(drop=True)

n, p = df_clean.shape[0], len(bien_chinh)
print(f"Kích thước sau làm sạch: n = {n}, p = {p}")

def phan_loai(diq):
    if diq == 2: return 0
    elif diq == 3: return 1
    elif diq == 1: return 2
    else: return np.nan

df_clean["Severity"] = df_clean["DIQ010"].apply(phan_loai)

# Lưu dữ liệu gốc (sau merge, trước làm sạch)
df_raw.to_csv("nhanes_raw.csv", index=False, encoding="utf-8-sig")

# Lưu dữ liệu đã làm sạch (đã có Severity + D2 + outlier flag)
df_clean.to_csv("nhanes_clean.csv", index=False, encoding="utf-8-sig")

print("\nSố lượng quan sát theo từng nhóm Severity:")
print(df_clean["Severity"].value_counts().sort_index())


X = df_clean[bien_chinh].to_numpy(dtype=float)
x_bar = X.mean(axis=0)
S = np.cov(X, rowvar=False, ddof=1)

D2 = np.zeros(n)
for i in range(n):
    d = X[i] - x_bar
    z = np.linalg.solve(S, d)
    D2[i] = d @ z

df_clean["D2"] = D2
nguong = chi2.ppf(0.975, df=p)
df_clean["Nghi_ngo_Outlier"] = df_clean["D2"] > nguong

print(f"\nNgưỡng cảnh báo Mahalanobis: {nguong:.4f}")
print(f"Số quan sát bị gắn cờ: {df_clean['Nghi_ngo_Outlier'].sum()}")


# 3.2. Tính các đặc trưng mấu
# ---------------------------
print("\nVector trung bình tổng thể:")
print(pd.Series(x_bar, index=bien_chinh).round(4))

print("\nVector trung bình theo từng nhóm:")
print(df_clean.groupby("Severity")[bien_chinh].mean().round(3))

Xc = X - x_bar
S_thu_cong = (Xc.T @ Xc) / (n - 1)
print("\nSai lệch cực đại giữa 2 cách tính S:", np.max(np.abs(S - S_thu_cong)))
print("Khớp hoàn toàn?", np.allclose(S, S_thu_cong))

S_df = pd.DataFrame(S, index=bien_chinh, columns=bien_chinh)
print("\nMa trận hiệp phương sai S:")
print(S_df.round(4))

R = np.corrcoef(X, rowvar=False)
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S)))
R_thu_cong = D_inv_sqrt @ S @ D_inv_sqrt
print("\nSai lệch cực đại giữa 2 cách tính R:", np.max(np.abs(R - R_thu_cong)))
print("Khớp hoàn toàn?", np.allclose(R, R_thu_cong))

R_df = pd.DataFrame(R, index=bien_chinh, columns=bien_chinh)
print("\nMa trận tương quan R:")
print(R_df.round(4))

det_S = np.linalg.det(S)
trace_S = np.trace(S)
hang_S = np.linalg.matrix_rank(S)
tri_rieng = np.linalg.eigvalsh(S)
cond_S = np.linalg.cond(S)

print(f"\nĐịnh thức |S|      : {det_S:.6e}")
print(f"Vết tr(S)            : {trace_S:.4f}")
print(f"Hạng r               : {hang_S} / p = {p}")
print(f"Số điều kiện cond(S) : {cond_S:.4f}")
print(f"Các trị riêng của S  : {np.round(tri_rieng, 4)}")
print(f"S xác định dương?    : {np.all(tri_rieng > 0)}")


# 3.3. Kiểm đinh giả định thống kê
# --------------------------------

hz = pg.multivariate_normality(X, alpha=0.05)
print("\n=== KIỂM ĐỊNH HENZE-ZIRKLER ===")
print(hz)

D2_sx = np.sort(df_clean["D2"].to_numpy())
vi_tri = (np.arange(1, n + 1) - 0.5) / n
phan_vi_ly_thuyet = chi2.ppf(vi_tri, df=p)
r_qq = np.corrcoef(D2_sx, phan_vi_ly_thuyet)[0, 1]

plt.figure(figsize=(7, 5))
plt.scatter(phan_vi_ly_thuyet, D2_sx, s=40, color="#0056A6",
            edgecolors="k", alpha=0.7, label=f"Dữ liệu (r = {r_qq:.4f})")
gia_tri_max = max(phan_vi_ly_thuyet.max(), D2_sx.max()) + 1
plt.plot([0, gia_tri_max], [0, gia_tri_max], "--",
         color="#F57C00", linewidth=2, label=f"Đường lý thuyết chi2_{p}")
plt.title("Chi-Square Q-Q Plot kiểm tra chuẩn nhiều chiều",
          fontsize=12, fontweight="bold")
plt.xlabel(f"Phân vị lý thuyết chi2_{p}")
plt.ylabel("Khoảng cách Mahalanobis D2 có thứ tự")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

print(f"\nHệ số tương quan Q-Q: r = {r_qq:.4f}")

box_m = pg.box_m(data=df_clean, dvs=bien_chinh, group="Severity")
print("\n=== KIỂM ĐỊNH BOX'S M ===")
print(box_m)


# 3.4. Kiểm định Manova
# ---------------------
formula = " + ".join(bien_chinh) + " ~ Severity"
print(f"\nCông thức MANOVA: {formula}\n")

maov = MANOVA.from_formula(formula, data=df_clean)
ket_qua = maov.mv_test()
print(ket_qua)

cac_nhom = sorted(df_clean["Severity"].unique())
g = len(cac_nhom)
df1 = p * (g - 1)
df2 = n - g - p + 1
alpha = 0.05
F_crit = f_dist.ppf(1 - alpha, df1, df2)

print(f"\nGiá trị tới hạn F({df1}, {df2}) ở mức 0.05: {F_crit:.4f}")

ket_qua_anova = []
for col in bien_chinh:
    du_lieu_nhom = [df_clean[df_clean["Severity"] == g_][col].to_numpy()
                    for g_ in cac_nhom]
    F_val, p_val = stats.f_oneway(*du_lieu_nhom)
    ket_qua_anova.append({"Biến": col, "F": F_val, "p_raw": p_val})

df_anova = pd.DataFrame(ket_qua_anova)

reject, p_adj, _, _ = multipletests(df_anova["p_raw"], alpha=0.05, method="holm")
df_anova["p_Holm"] = p_adj
df_anova["Bác bỏ H0"] = reject

print("\n=== FOLLOW-UP ANOVA + HIỆU CHỈNH HOLM ===")
print(df_anova.round(4))

# Code Post-hoc
from itertools import combinations
from scipy.stats import ttest_ind

pairs = list(combinations(cac_nhom, 2))
posthoc_rows = []

for col in bien_chinh:
    for a, b in pairs:
        g1 = df_clean[df_clean["Severity"] == a][col].to_numpy()
        g2 = df_clean[df_clean["Severity"] == b][col].to_numpy()
        t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
        posthoc_rows.append({
            "Biến": col, "Cặp nhóm": f"{a} vs {b}",
            "Chênh lệch TB": round(g1.mean() - g2.mean(), 3),
            "t_stat": round(t_stat, 4),
            "p_raw": p_val
        })

df_posthoc = pd.DataFrame(posthoc_rows)
reject, p_adj, _, _ = multipletests(df_posthoc["p_raw"],
                                    alpha=0.05, method="holm")
df_posthoc["p_Holm"] = p_adj
df_posthoc["Bác bỏ H0"] = reject
print(df_posthoc.round(4))