from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

n = 4000
age = np.random.randint(18, 75, n)
tenure = np.random.randint(1, 60, n)
usage = np.random.gamma(shape=2.0, scale=50.0, size=n)
issues = np.random.poisson(lam=0.3, size=n)
is_promo = np.random.binomial(1, 0.25, n)

logit = -3.0 + 0.02*(75-age) - 0.03*tenure - 0.002*usage + 0.5*issues - 0.6*is_promo
prob = 1/(1+np.exp(-logit))
churn = np.random.binomial(1, prob)

df = pd.DataFrame({'age':age,'tenure':tenure,'usage':usage,'issues':issues,'is_promo':is_promo,'churn':churn})

X = df.drop(columns=['churn'])
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

num_cols = ['age','tenure','usage','issues']
scaler = StandardScaler().fit(X_train[num_cols])
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

proba_lr = lr.predict_proba(X_test_scaled)[:,1]
proba_rf = rf.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(6,5))
plt.plot(fpr_lr, tpr_lr, label=f'LogReg AUC={auc_lr:.3f}')
plt.plot(fpr_rf, tpr_rf, label=f'RandomForest AUC={auc_rf:.3f}')
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC — Churn Prediction')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc.png', dpi=140)
plt.close()

# Отчёты
from sklearn.metrics import classification_report
pred_lr = (proba_lr >= 0.5).astype(int)
pred_rf = (proba_rf >= 0.5).astype(int)
pd.DataFrame(classification_report(y_test, pred_lr, output_dict=True)).to_csv(OUTPUT_DIR / 'classification_report_logreg.csv')
pd.DataFrame(classification_report(y_test, pred_rf, output_dict=True)).to_csv(OUTPUT_DIR / 'classification_report_rf.csv')
df.to_csv(OUTPUT_DIR / 'synthetic_churn_data.csv', index=False)

print("✅ Churn-моделирование завершено. Артефакты в", OUTPUT_DIR)