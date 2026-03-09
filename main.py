"""
Logistic Regression Comprehensive Demo
Covers binary/multiclass classification, sigmoid, decision boundaries, ROC, odds ratios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_sigmoid(save_path='sigmoid_function.png'):
    """Illustrate the sigmoid function."""
    z = np.linspace(-10, 10, 200)
    p = sigmoid(z)
    plt.figure(figsize=(8, 5))
    plt.plot(z, p, 'b-', lw=2)
    plt.axhline(0.5, color='r', linestyle='--', alpha=0.7, label='p=0.5 (threshold)')
    plt.axvline(0,   color='g', linestyle='--', alpha=0.7, label='z=0')
    plt.fill_between(z[z >= 0], p[z >= 0], 0.5, alpha=0.1, color='green', label='Predict 1')
    plt.fill_between(z[z <= 0], p[z <= 0], 0.5, alpha=0.1, color='red',   label='Predict 0')
    plt.xlabel('Linear combination z = w·x + b')
    plt.ylabel('Probability σ(z)')
    plt.title('Sigmoid (Logistic) Function')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sigmoid plot saved to {save_path}")


def demo_binary_classification():
    """Binary classification with decision boundary."""
    print("\n" + "="*50)
    print("1. BINARY LOGISTIC REGRESSION")
    print("="*50)
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=1.0, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    print(f"  Accuracy={acc:.4f}, AUC={auc:.4f}")
    print(f"  Coefficients: {model.coef_[0]}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")

    # Decision boundary
    h = 0.05
    x1_min, x1_max = X_s[:, 0].min() - 1, X_s[:, 0].max() + 1
    x2_min, x2_max = X_s[:, 1].min() - 1, X_s[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = model.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1].reshape(xx1.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cs = axes[0].contourf(xx1, xx2, Z, levels=50, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(cs, ax=axes[0], label='P(class=1)')
    axes[0].scatter(X_s[y==0, 0], X_s[y==0, 1], c='blue',  s=20, alpha=0.5, label='Class 0')
    axes[0].scatter(X_s[y==1, 0], X_s[y==1, 1], c='red',   s=20, alpha=0.5, label='Class 1')
    axes[0].set_title('Decision Boundary'); axes[0].legend()

    fpr, tpr, _ = roc_curve(y_te, y_prob)
    axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'AUC={auc:.3f}')
    axes[1].plot([0,1],[0,1],'k--')
    axes[1].set_title('ROC Curve'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('binary_classification.png', dpi=150)
    plt.close()
    return model


def demo_multiclass():
    """Multinomial logistic regression."""
    print("\n" + "="*50)
    print("2. MULTICLASS LOGISTIC REGRESSION")
    print("="*50)
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=42)

    for solver, multi in [('lbfgs','multinomial'), ('lbfgs','ovr')]:
        model = LogisticRegression(solver=solver, multi_class=multi, C=1.0, random_state=42)
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        cv  = cross_val_score(model, X_s, y, cv=5, scoring='accuracy').mean()
        print(f"  {multi.upper():12s}: Test={acc:.4f}, CV={cv:.4f}")

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1.0, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(classification_report(y_te, y_pred))

    # Plot
    h = 0.1
    xx, yy = np.meshgrid(np.arange(X_s[:,0].min()-1, X_s[:,0].max()+1, h),
                         np.arange(X_s[:,1].min()-1, X_s[:,1].max()+1, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='Set3')
    colors = ['red','blue','green','purple']
    for cls in range(4):
        mask = y == cls
        plt.scatter(X_s[mask, 0], X_s[mask, 1], c=colors[cls], s=25, alpha=0.7, label=f'Class {cls}')
    plt.title('Multiclass Logistic Regression')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiclass.png', dpi=150)
    plt.close()


def demo_regularization_effect():
    """Show effect of C (inverse regularization) on decision boundary."""
    print("\n" + "="*50)
    print("3. REGULARIZATION STRENGTH")
    print("="*50)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=0)

    C_values = [0.001, 0.1, 1.0, 10, 100]
    fig, axes = plt.subplots(1, len(C_values), figsize=(18, 4))
    for ax, C in zip(axes, C_values):
        m = LogisticRegression(C=C).fit(X_tr, y_tr)
        acc = m.score(X_te, y_te)
        xx, yy = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
        Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
        ax.scatter(X_s[:,0], X_s[:,1], c=y, cmap='RdBu', s=15, alpha=0.7)
        ax.set_title(f'C={C}\nacc={acc:.2f}', fontsize=9)
        ax.set_xlim(-3,3); ax.set_ylim(-3,3)
        ax.axis('off')
    plt.suptitle('Effect of Regularization (C) on Decision Boundary')
    plt.tight_layout()
    plt.savefig('regularization_effect.png', dpi=150)
    plt.close()
    print("  Regularization plot saved.")


def demo_odds_ratios():
    """Compute and visualize odds ratios for interpretability."""
    print("\n" + "="*50)
    print("4. ODDS RATIOS & INTERPRETABILITY")
    print("="*50)
    np.random.seed(42)
    n = 1000
    age        = np.random.normal(45, 15, n)
    income     = np.random.normal(50000, 20000, n)
    debt_ratio = np.random.uniform(0, 1, n)
    employed   = np.random.binomial(1, 0.7, n)
    logit      = -2 + 0.03*age + 0.00001*income - 3*debt_ratio + 1.5*employed
    y          = np.random.binomial(1, sigmoid(logit), n)

    df = pd.DataFrame({'age': age, 'income': income, 'debt_ratio': debt_ratio,
                        'employed': employed})
    scaler = StandardScaler()
    X_s = scaler.fit_transform(df)
    model = LogisticRegression(C=10).fit(X_s, y)

    odds_ratios = np.exp(model.coef_[0])
    feat_names  = df.columns.tolist()
    print("  Feature       | Coefficient | Odds Ratio | Interpretation")
    print("  " + "-"*60)
    for f, coef, OR in zip(feat_names, model.coef_[0], odds_ratios):
        direction = 'increases' if coef > 0 else 'decreases'
        print(f"  {f:13s} | {coef:+.4f}     | {OR:8.3f}   | 1-std increase {direction} odds by {abs(OR-1):.1%}")

    plt.figure(figsize=(8, 5))
    colors = ['green' if c > 0 else 'red' for c in model.coef_[0]]
    bars = plt.barh(feat_names, odds_ratios - 1, color=colors, alpha=0.7)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Odds Ratio - 1')
    plt.title('Feature Effect on Odds (Log Odds = Coefficient)')
    plt.tight_layout()
    plt.savefig('odds_ratios.png', dpi=150)
    plt.close()
    print("  Odds ratio plot saved.")


def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION COMPREHENSIVE DEMO")
    print("=" * 60)

    plot_sigmoid()
    demo_binary_classification()
    demo_multiclass()
    demo_regularization_effect()
    demo_odds_ratios()

    print("\n--- Output Files ---")
    for f in ['sigmoid_function.png', 'binary_classification.png',
              'multiclass.png', 'regularization_effect.png', 'odds_ratios.png']:
        print(f"  {f}")

    print("\n✓ Logistic Regression Demo complete!")


if __name__ == '__main__':
    main()
