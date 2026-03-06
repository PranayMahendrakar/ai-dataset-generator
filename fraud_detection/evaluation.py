"""
Model Evaluation and Visualization
====================================
ROC/PR curves, confusion matrices, feature importance, score distribution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import os

COLORS = {
    'isolation_forest': '#E74C3C',
    'random_forest': '#3498DB',
    'xgboost': '#2ECC71',
    'fraud': '#E74C3C',
    'legit': '#3498DB',
    'bg': '#0D1117',
    'text': '#C9D1D9',
    'grid': '#21262D',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['bg'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid'],
})


def _save(fig, path, dpi=150):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    return path


def plot_roc_curves(y_true, scores, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('ROC Curves', fontsize=14, fontweight='bold')
    for name, proba in scores.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name.replace('_',' ').title()} AUC={auc:.3f}",
                color=COLORS.get(name, '#AAA'), linewidth=2)
    ax.plot([0,1],[0,1],'--',color='#555',lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    return _save(fig, save_path) if save_path else fig


def plot_pr_curves(y_true, scores, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.axhline(y_true.mean(), color='#555', ls='--', lw=1, label='Baseline')
    for name, proba in scores.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(r, p, label=f"{name.replace('_',' ').title()} AP={ap:.3f}",
                color=COLORS.get(name, '#AAA'), linewidth=2)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.legend(); ax.grid(True, alpha=0.3)
    return _save(fig, save_path) if save_path else fig


def plot_confusion_matrices(y_true, predictions, save_path=None):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    if n == 1: axes = [axes]
    for ax, (name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, preds)
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        fontsize=18, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()/2 else COLORS['text'])
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Legit','Fraud']); ax.set_yticklabels(['Legit','Fraud'])
        ax.set_title(name.replace('_',' ').title())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _save(fig, save_path) if save_path else fig


def plot_feature_importance(feature_names, importances,
                             model_name='Random Forest', top_n=15, save_path=None):
    idx = np.argsort(importances)[-top_n:]
    feats = [feature_names[i].replace('_',' ') for i in idx]
    vals = importances[idx]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n*0.45)))
    fig.suptitle(f'Feature Importance - {model_name}', fontsize=13, fontweight='bold')
    bars = ax.barh(feats, vals, color=COLORS['random_forest'], alpha=0.85)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xlabel('Importance'); ax.grid(True, axis='x', alpha=0.3)
    return _save(fig, save_path) if save_path else fig


def plot_score_distribution(combined_scores, y_true, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Fraud Score Distribution', fontsize=14, fontweight='bold')
    bins = np.linspace(0, 1, 50)
    ax.hist(combined_scores[y_true==0], bins=bins, alpha=0.55,
            color=COLORS['legit'], label='Legitimate', density=True)
    ax.hist(combined_scores[y_true==1], bins=bins, alpha=0.55,
            color=COLORS['fraud'], label='Fraud', density=True)
    ax.axvline(0.5, color='#F39C12', ls='--', lw=1.5, label='Decision=0.5')
    ax.set_xlabel('Fraud Probability'); ax.set_ylabel('Density')
    ax.legend(); ax.grid(True, alpha=0.3)
    return _save(fig, save_path) if save_path else fig


def plot_dashboard(y_true, scores, predictions, feature_names, rf_importances,
                   save_path='fraud_detection/output/dashboard.png'):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Credit Card Fraud Detection Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0,0])
    for name, proba in scores.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax1.plot(fpr, tpr, label=f"{name.split('_')[0].title()} {auc:.3f}",
                 color=COLORS.get(name,'#AAA'), lw=2)
    ax1.plot([0,1],[0,1],'--',color='#555',lw=1)
    ax1.set_title('ROC Curves'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0,1])
    for name, proba in scores.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax2.plot(r, p, label=f"{name.split('_')[0].title()} AP={ap:.3f}",
                 color=COLORS.get(name,'#AAA'), lw=2)
    ax2.axhline(y_true.mean(), color='#555', ls='--', lw=1)
    ax2.set_title('Precision-Recall'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0,2])
    xpreds = predictions.get('xgboost', list(predictions.values())[-1])
    cm = confusion_matrix(y_true, xpreds)
    im = ax3.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=16, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else COLORS['text'])
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(['Legit','Fraud']); ax3.set_yticklabels(['Legit','Fraud'])
    ax3.set_title('Confusion Matrix (XGBoost)')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(gs[1,0])
    top = 12
    idx = np.argsort(rf_importances)[-top:]
    feats = [feature_names[i].replace('_',' ') for i in idx]
    vals = rf_importances[idx]
    ax4.barh(feats, vals, color=COLORS['random_forest'], alpha=0.85)
    ax4.set_title(f'Feature Importance (RF top-{top})')
    ax4.tick_params(axis='y', labelsize=8); ax4.grid(True, axis='x', alpha=0.3)

    ax5 = fig.add_subplot(gs[1,1])
    ensemble = sum(scores.values()) / len(scores)
    bins = np.linspace(0, 1, 50)
    ax5.hist(ensemble[y_true==0], bins=bins, alpha=0.55, color=COLORS['legit'],
             label='Legit', density=True)
    ax5.hist(ensemble[y_true==1], bins=bins, alpha=0.55, color=COLORS['fraud'],
             label='Fraud', density=True)
    ax5.axvline(0.5, color='#F39C12', ls='--', lw=1.5, label='Threshold')
    ax5.set_title('Score Distribution'); ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1,2])
    mnames = list(scores.keys())
    aucs = [roc_auc_score(y_true, scores[m]) for m in mnames]
    aps  = [average_precision_score(y_true, scores[m]) for m in mnames]
    x = np.arange(len(mnames)); w = 0.35
    b1 = ax6.bar(x-w/2, aucs, w, label='ROC-AUC', color=COLORS['random_forest'], alpha=0.85)
    b2 = ax6.bar(x+w/2, aps,  w, label='Avg Prec', color=COLORS['xgboost'], alpha=0.85)
    ax6.bar_label(b1, fmt='%.3f', padding=2, fontsize=8)
    ax6.bar_label(b2, fmt='%.3f', padding=2, fontsize=8)
    ax6.set_xticks(x)
    ax6.set_xticklabels([m.replace('_','\n') for m in mnames], fontsize=8)
    ax6.set_ylim(0, 1.15); ax6.set_title('Model Comparison')
    ax6.legend(); ax6.grid(True, axis='y', alpha=0.3)

    d = os.path.dirname(save_path)
    if d: os.makedirs(d, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    return save_path


def evaluate_and_plot(detector, X_test, y_test, feature_names,
                       output_dir='fraud_detection/output'):
    os.makedirs(output_dir, exist_ok=True)
    scores = {
        'isolation_forest': detector.anomaly_detector.score(X_test),
        'random_forest':    detector.rf_model.predict_proba(X_test),
        'xgboost':          detector.xgb_model.predict_proba(X_test),
    }
    preds = {
        'isolation_forest': detector.anomaly_detector.predict(X_test),
        'random_forest':    detector.rf_model.predict(X_test),
        'xgboost':          detector.xgb_model.predict(X_test),
    }
    paths = {
        'roc':       plot_roc_curves(y_test, scores, f'{output_dir}/roc_curves.png'),
        'pr':        plot_pr_curves(y_test, scores, f'{output_dir}/pr_curves.png'),
        'confusion': plot_confusion_matrices(y_test, preds, f'{output_dir}/confusion_matrices.png'),
        'importance': plot_feature_importance(
                          feature_names, detector.rf_model.feature_importances_,
                          save_path=f'{output_dir}/feature_importance.png'),
        'scores':    plot_score_distribution(
                          sum(scores.values())/len(scores), y_test,
                          save_path=f'{output_dir}/score_distribution.png'),
        'dashboard': plot_dashboard(y_test, scores, preds, feature_names,
                          detector.rf_model.feature_importances_,
                          save_path=f'{output_dir}/dashboard.png'),
    }
    print(f"All plots saved to: {output_dir}")
    return {'paths': paths, 'scores': scores, 'predictions': preds}
