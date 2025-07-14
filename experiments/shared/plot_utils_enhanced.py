"""
Enhanced plot utilities that support configuration from experiments.yaml
This extends the existing plot_utils.py without breaking compatibility
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List, Union

# Import existing utilities
from . import plot_utils

logger = logging.getLogger(__name__)

def save_plot_multi_format(fig, plot_dir: Path, plot_name: str, 
                          dpi: int = 300, 
                          bbox_inches: str = 'tight',
                          formats: Optional[List[str]] = None):
    """
    Enhanced save function that supports multiple formats
    
    Args:
        fig: Matplotlib figure object
        plot_dir: Directory to save plots in
        plot_name: Base filename (extension will be added)
        dpi: DPI for raster formats
        bbox_inches: Bounding box setting
        formats: List of formats to save (e.g., ['png', 'pdf'])
    """
    if formats is None:
        formats = ['png']
    
    if not plot_dir.is_dir():
        logger.error(f"Plot directory does not exist: {plot_dir}")
        return
    
    # Remove any existing extension from plot_name
    base_name = plot_name.rsplit('.', 1)[0] if '.' in plot_name else plot_name
    
    for fmt in formats:
        save_path = plot_dir / f"{base_name}.{fmt}"
        try:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, format=fmt)
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save {fmt} plot {save_path}: {e}")
    
    plt.close(fig)

def apply_plot_config(config: Dict[str, Any], plot_type: Optional[str] = None):
    """
    Apply plotting configuration to matplotlib/seaborn
    
    Args:
        config: Plotting configuration dictionary from experiments.yaml
        plot_type: Optional specific plot type (e.g., 'regret_plot', 'heatmap')
    """
    # Get base config
    fonts = config.get('fonts', {})
    
    # If specific plot type config exists, merge it
    if plot_type and plot_type in config:
        plot_config = config[plot_type]
        if 'fonts' in plot_config:
            fonts = {**fonts, **plot_config['fonts']}
    
    # Apply font settings if any specified
    if fonts:
        rcParams = {}
        if 'base_size' in fonts:
            rcParams['font.size'] = fonts['base_size']
        if 'label_size' in fonts:
            rcParams['axes.labelsize'] = fonts['label_size']
        if 'tick_size' in fonts:
            rcParams['xtick.labelsize'] = fonts['tick_size']
            rcParams['ytick.labelsize'] = fonts['tick_size']
        if 'legend_size' in fonts:
            rcParams['legend.fontsize'] = fonts['legend_size']
        if 'title_size' in fonts:
            rcParams['axes.titlesize'] = fonts['title_size']
        
        if rcParams:
            plt.rcParams.update(rcParams)
            logger.debug(f"Applied rcParams: {rcParams}")

def get_figure_size(config: Dict[str, Any], plot_type: Optional[str] = None, 
                    default: tuple = None) -> tuple:
    """
    Get figure size from config
    
    Args:
        config: Plotting configuration
        plot_type: Optional specific plot type
        default: Default size if not specified
        
    Returns:
        Tuple of (width, height)
    """
    # Check plot-specific config first
    if plot_type and plot_type in config:
        plot_config = config[plot_type]
        if 'figsize' in plot_config:
            return tuple(plot_config['figsize'])
    
    # Check general figsize
    if 'figsize' in config:
        return tuple(config['figsize'])
    
    # Return default or None (let matplotlib decide)
    return default

def get_plot_value(config: Dict[str, Any], key: str, plot_type: Optional[str] = None, 
                   default: Any = None) -> Any:
    """
    Get a value from plot config with fallback logic
    
    Args:
        config: Plotting configuration
        key: Key to look for
        plot_type: Optional specific plot type
        default: Default value if not found
        
    Returns:
        The configured value or default
    """
    # Check plot-specific config first
    if plot_type and plot_type in config:
        plot_config = config[plot_type]
        if key in plot_config:
            return plot_config[key]
    
    # Check general config
    if key in config:
        return config[key]
    
    # Return default
    return default

# Maintain backward compatibility by re-exporting original functions
get_color = plot_utils.get_color
get_marker = plot_utils.get_marker
setup_plotting = plot_utils.setup_plotting
save_plot = plot_utils.save_plot  # Original single-format save