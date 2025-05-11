#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot metrics from RLHF training logs')
parser.add_argument('--metric', default='mean_r', help='Metric to plot (default: mean_r)')
parser.add_argument('--output', default=None, help='If provided, save the plot to this file')
parser.add_argument('paths', nargs='+', help='CSV log files to plot')
args = parser.parse_args()

# Create figure
plt.figure(figsize=(10, 6))

# Plot each log file
for p in args.paths:
    df = pd.read_csv(p)
    label = p.split('/')[-1].split('.')[0]
    linestyle = '-' if 'grpo' in label.lower() else '--'
    plt.plot(df['step'], df[args.metric], linestyle, label=label)

# Add labels and legend
plt.xlabel('Step')
plt.ylabel(args.metric.replace('_', ' ').title())
plt.title(f'Comparison of {args.metric.replace("_", " ").title()} between Models')
plt.legend()
plt.grid(True, alpha=0.3)

# Save or show the plot
if args.output:
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")
else:
    plt.show() 