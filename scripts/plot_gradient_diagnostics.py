import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the metrics
metrics_df = pd.read_csv("metrics.csv")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Gradient Analysis During Training', fontsize=16, fontweight='bold')

# 1. Cosine Similarity over steps/epochs (showing gradient conflicts)
ax1 = axes[0, 0]
ax1.plot(metrics_df['step'], metrics_df['train/grad_cosine'], 'b-', alpha=0.7, linewidth=1)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Orthogonal')
ax1.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='Aligned')
ax1.axhline(y=-1, color='r', linestyle='--', alpha=0.3, label='Conflicting')
ax1.fill_between(metrics_df['step'], 0, metrics_df['train/grad_cosine'], 
                  where=(metrics_df['train/grad_cosine'] > 0), 
                  color='green', alpha=0.3, label='Aligned (cos > 0)')
ax1.fill_between(metrics_df['step'], 0, metrics_df['train/grad_cosine'], 
                  where=(metrics_df['train/grad_cosine'] < 0), 
                  color='red', alpha=0.3, label='Conflicting (cos < 0)')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Gradient Cosine Similarity (JEPA vs Classifier)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 2. Gradient norms (magnitudes)
ax2 = axes[0, 1]
ax2.plot(metrics_df['step'], metrics_df['train/grad_norm_jepa'], 'b-', label='JEPA Gradient Norm', alpha=0.7)
ax2.plot(metrics_df['step'], metrics_df['train/grad_norm_cls'], 'r-', label='Classifier Gradient Norm', alpha=0.7)
ax2.set_yscale('log')
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Gradient Norm (log scale)')
ax2.set_title('Gradient Magnitudes')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Losses over time
ax3 = axes[1, 0]
ax3.plot(metrics_df['step'], metrics_df['train/loss'], 'k-', label='Total Loss', linewidth=1.5)
ax3.plot(metrics_df['step'], metrics_df['train/jepa_loss'], 'b-', label='JEPA Loss', alpha=0.7)
ax3.plot(metrics_df['step'], metrics_df['train/cls_loss'], 'r-', label='Classifier Loss', alpha=0.7)
ax3.set_xlabel('Training Step')
ax3.set_ylabel('Loss')
ax3.set_title('Training Losses')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Gradient conflict ratio (percentage of steps with negative cosine)
ax4 = axes[1, 1]
window = 10  # rolling window for smoothing
conflicts = (metrics_df['train/grad_cosine'] < 0).astype(int)
conflict_ratio = conflicts.rolling(window=window, min_periods=1).mean() * 100

ax4.plot(metrics_df['step'], conflict_ratio, 'r-', linewidth=1.5)
ax4.fill_between(metrics_df['step'], 0, conflict_ratio, alpha=0.3, color='red')
ax4.set_xlabel('Training Step')
ax4.set_ylabel(f'Conflict Ratio (% of steps with cos < 0)\n({window}-step rolling window)')
ax4.set_title('Gradient Conflict Frequency')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("GRADIENT CONFLICT SUMMARY")
print("="*60)
print(f"Total training steps analyzed: {len(metrics_df)}")
print(f"Steps with positive cosine (aligned): {(metrics_df['train/grad_cosine'] > 0).sum()} ({(metrics_df['train/grad_cosine'] > 0).mean()*100:.1f}%)")
print(f"Steps with negative cosine (conflicting): {(metrics_df['train/grad_cosine'] < 0).sum()} ({(metrics_df['train/grad_cosine'] < 0).mean()*100:.1f}%)")
print(f"Average cosine similarity: {metrics_df['train/grad_cosine'].mean():.4f}")
print(f"Median cosine similarity: {metrics_df['train/grad_cosine'].median():.4f}")
print(f"Std of cosine similarity: {metrics_df['train/grad_cosine'].std():.4f}")
print(f"\nGradient Norm Statistics:")
print(f"  JEPA - Mean: {metrics_df['train/grad_norm_jepa'].mean():.2f}, Std: {metrics_df['train/grad_norm_jepa'].std():.2f}")
print(f"  Classifier - Mean: {metrics_df['train/grad_norm_cls'].mean():.2f}, Std: {metrics_df['train/grad_norm_cls'].std():.2f}")
print(f"  Norm Ratio (Classifier/JEPA): {(metrics_df['train/grad_norm_cls'] / metrics_df['train/grad_norm_jepa']).mean():.2f}")

# Check if gradients are conflicting more at the start or end
early_steps = metrics_df.head(20)
late_steps = metrics_df.tail(20)
print(f"\nEarly training (first 20 steps) - Cosine mean: {early_steps['train/grad_cosine'].mean():.4f}")
print(f"Late training (last 20 steps) - Cosine mean: {late_steps['train/grad_cosine'].mean():.4f}")

if early_steps['train/grad_cosine'].mean() > late_steps['train/grad_cosine'].mean():
    print("\n?? Gradients are becoming MORE conflicting over time")
else:
    print("\n?? Gradients are becoming LESS conflicting over time")

# Highlight potential issues
if metrics_df['train/grad_cosine'].mean() < 0:
    print("\n??  WARNING: Average gradient conflict is negative!")
    print("   JEPA and classifier gradients are generally opposing each other.")
    print("   Consider adjusting lambda_cls or using different optimization strategy.")
else:
    print("\n? Average gradient alignment is positive.")