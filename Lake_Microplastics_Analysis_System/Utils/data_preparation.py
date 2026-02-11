# s14_data_preparation.py
# Final version with professional, editable font settings.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_environment():
    """Configures the environment with paths and publication-quality plot settings."""
    data_path = r"E:\lake-MP-W\data\train\train_data.csv"
    output_dir = r"E:\lake-MP-W\draw"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' created or already exists.")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{data_path}'")
        return None, None

    df['mp_abundance'] = np.exp(df['ln'])

    # --- Publication-Quality Font and Editability Settings ---
    sns.set_theme(style="white", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['pdf.fonttype'] = 42  # Ensures text is editable in Adobe Illustrator
    plt.rcParams['svg.fonttype'] = 'none'  # Ensures text is editable in SVG
    plt.rcParams['axes.unicode_minus'] = False

    print("Plotting environment configured with Arial font for editable PDF output.")
    return df, output_dir


def save_plot(fig, filename, output_dir):
    """Saves the figure in high-quality PNG and editable PDF formats."""
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to '{png_path}' and '{pdf_path}'")
    plt.close(fig)