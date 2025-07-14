
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

from experiments.shared import plot_utils

# Logger configuration
logger = logging.getLogger(__name__)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def generate_reward_barplot(summary_stats: pd.DataFrame, exp_dirs: dict, title: str = "Average Reward by Feature Set"):
    """Generates a bar plot comparing average reward across feature sets."""
    logger.info("Generating A3 Feature Ablation reward bar plot...")

    if summary_stats.empty:
        logger.warning("Summary statistics DataFrame is empty. Cannot generate bar plot.")
        return

    required_cols = ['feature_set', 'mean_reward'] 
    if not all(col in summary_stats.columns for col in required_cols):
        logger.error(f"Missing required columns for bar plot: {required_cols}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [plot_utils.get_color(fs) for fs in summary_stats['feature_set']]
    bars = ax.bar(
        summary_stats['feature_set'],
        summary_stats['mean_reward'],
        color=colors, 
        alpha=0.8
    )

    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Mean Reward") 
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    if len(summary_stats['feature_set']) > 5:
        ax.tick_params(axis='x', rotation=30, ha='right')
    else:
         ax.tick_params(axis='x', rotation=0)
    min_val = 0
    max_val = summary_stats['mean_reward'].max()
    padding = max_val * 0.1
    ax.set_ylim(min_val, max_val + padding) 

    plt.tight_layout()

    plot_filename = f"{exp_dirs['base'].name}_reward_barplot.png"
    plot_utils.save_plot(fig, exp_dirs['plots'], plot_filename)
def generate_regret_boxplot(final_regret_df: pd.DataFrame, exp_dirs: dict, title: str = "Final Cumulative Regret Distribution",
                            category_order: list = None):
    """Generates a box plot showing the distribution of final cumulative regret per feature set."""

    logger.info("Generating A3 final cumulative regret box plot...")

    if final_regret_df.empty:
        logger.warning("Final regret DataFrame is empty. Cannot generate box plot.")
        return

    required_cols = ['feature_set', 'cumulative_regret']
    if not all(col in final_regret_df.columns for col in required_cols):
        logger.error(f"Missing required columns for regret box plot: {required_cols}")
        return
    if category_order:
        plot_order = category_order
        if not pd.api.types.is_categorical_dtype(final_regret_df['feature_set']):
            final_regret_df['feature_set'] = pd.Categorical(final_regret_df['feature_set'], categories=plot_order, ordered=True)
        final_regret_df = final_regret_df.sort_values('feature_set')
    else:
        logger.warning("No category_order provided to generate_regret_boxplot. Using alphabetical order.")
        plot_order = sorted(final_regret_df['feature_set'].unique())
        if not pd.api.types.is_categorical_dtype(final_regret_df['feature_set']):
             final_regret_df['feature_set'] = pd.Categorical(final_regret_df['feature_set'], categories=plot_order, ordered=True)
             final_regret_df = final_regret_df.sort_values('feature_set')
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        'none': '#d3d3d3',
        'single': '#1f77b4',
        'double': '#ff7f0e',
        'full': '#2ca02c' 
    }
    
    palette = []
    for fs in plot_order:
        if fs == 'none':
            palette.append(color_map['none'])
        elif fs == 'full':
            palette.append(color_map['full'])
        elif '_' in fs:
            palette.append(color_map['double'])
        else:
            palette.append(color_map['single'])

    sns.boxplot(
        data=final_regret_df,
        x='feature_set',
        y='cumulative_regret',
        order=plot_order,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Feature Set Used")
    ax.set_ylabel("Cumulative Regret")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plot_filename = f"{exp_dirs['base'].name}_final_regret_boxplot.png"
    plot_utils.save_plot(fig, exp_dirs['plots'], plot_filename)
