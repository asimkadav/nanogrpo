#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot metrics from RLHF training logs')
parser.add_argument('--metric', default='mean_r', help='Metric to plot (default: mean_r)')
parser.add_argument('--output', default=None, help='If provided, save the plot to this file')
parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figure (default: 300)')
parser.add_argument('paths', nargs='+', help='CSV log files to plot')
args = parser.parse_args()

# Create figure
plt.figure(figsize=(10, 6))

# Plot each log file
for p in args.paths:
    df = pd.read_csv(p)
    label = p.split('/')[-1].split('.')[0]
    linestyle = '-' if 'grpo' in label.lower() else '--'
    plt.plot(df['step'], df[args.metric], linestyle, label=label.upper(), linewidth=2)

# Map metric to more descriptive labels
ylabel_map = {
    'mean_r': 'Mean Reward',
    'loss': 'Loss Value',
    'kl': 'KL Divergence',
    'entropy': 'Entropy (Lower Bound)',
    'token_adv': 'Token Advantage',
}

# Add labels and legend with improved styling
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel(ylabel_map.get(args.metric, args.metric.replace('_', ' ').title()), fontsize=12)
plt.title(f'Comparison of {args.metric.replace("_", " ").title()} between Models', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Add a note for entropy if that's the metric
if args.metric == 'entropy':
    plt.figtext(0.5, 0.01, 'Note: Entropy values represent a lower bound (sampled-token only)', 
                ha='center', fontsize=8, style='italic')

# Save or show the plot
if args.output:
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Plot saved to {args.output} (DPI: {args.dpi})")
else:
    plt.show() 