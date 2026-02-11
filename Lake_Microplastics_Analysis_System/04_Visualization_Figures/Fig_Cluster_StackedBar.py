import pandas as pd
import matplotlib.pyplot as plt


def plot_chart_on_ax(ax, df, target_countries, ordered_countries, color_palette, chart_title):
    """
    A reusable function to draw a 100% stacked bar chart on a specific subplot axis.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis object to draw on.
        df (pd.DataFrame): The main DataFrame containing all data.
        target_countries (list): A list of country names for this specific chart.
        ordered_countries (list): The desired display order for countries.
        color_palette (list): The list of hex colors for clusters.
        chart_title (str): The title for this specific subplot.
    """
    # 1. Filter and process data for the current chart
    df_filtered = df[df['country'].isin(target_countries)].copy()

    if df_filtered.empty:
        ax.text(0.5, 0.5, 'No data available for this group.', horizontalalignment='center', verticalalignment='center')
        ax.set_title(chart_title, fontsize=20, pad=20, fontname='Arial', weight='bold')
        ax.set_axis_off()
        return

    country_cluster_counts = pd.crosstab(df_filtered['country'], df_filtered['cluster'])
    country_cluster_percentage = country_cluster_counts.div(country_cluster_counts.sum(axis=1), axis=0) * 100
    country_cluster_percentage = country_cluster_percentage.reindex(ordered_countries).dropna(how='all')

    # 2. Dynamic color mapping
    unique_clusters_in_data = sorted(country_cluster_percentage.columns.unique())
    cluster_color_map = {
        cluster_id: color_palette[i % len(color_palette)]
        for i, cluster_id in enumerate(unique_clusters_in_data)
    }
    plot_colors = [cluster_color_map.get(cluster_id, '#808080') for cluster_id in country_cluster_percentage.columns]

    # 3. Plot on the provided axis 'ax'
    country_cluster_percentage.plot(kind='bar', stacked=True, color=plot_colors, ax=ax, width=0.8, legend=False)

    # 4. Apply aesthetics to the provided axis 'ax'
    ax.set_title(chart_title, fontsize=20, pad=20, fontname='Arial', weight='bold')
    ax.set_ylabel('Cluster Percentage (%)', fontsize=16, labelpad=15, fontname='Arial')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14, fontname='Arial')
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)


def main():
    """
    Main function to create a single PDF with two horizontally-aligned charts.
    """
    # --- Global Configuration ---
    file_path = r"E:\lake-MP-W\draw\03_SHAP_Analysis\data_country_cluster.csv"
    output_filename = "cluster_distribution_combined.pdf"

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    map_script_colors = [
        "#4885c1", "#c96734", "#6aa4bb", "#e4cb3a",
        "#8c5374", "#aab381", "#ffc000", "#ae3a4e",
        "#afe1af", "#cad675", "#8dc2b5", "#6a6c9b"
    ]

    # --- Load Data Once ---
    try:
        df_main = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The main data file was not found at '{file_path}'. Exiting.")
        return

    # --- Define Country Lists (UPDATED as per your request) ---

    # Chart 1: Key Regions (Countries removed)
    target_1 = [
        'India', 'Pakistan', 'China', 'Vietnam', 'Cambodia', 'Laos',
        'Thailand', 'Myanmar (Burma)', 'Malaysia', 'Singapore', 'Indonesia',
        'Iraq', 'Nigeria', 'Brazil', 'Brunei'
    ]
    order_1 = [
        'India', 'Pakistan', 'China', 'Thailand', 'Vietnam', 'Laos', 'Cambodia',
        'Myanmar (Burma)', 'Malaysia', 'Singapore', 'Brunei', 'Indonesia',
        'Iraq', 'Nigeria', 'Brazil'
    ]

    # Chart 2: Selected Developed Nations (Countries removed)
    target_2 = ['Canada', 'Finland', 'Sweden', 'Norway', 'New Zealand']
    order_2 = ['Canada', 'Norway', 'Sweden', 'Finland', 'New Zealand']

    # --- Create a Single Figure with Two Horizontally Aligned Subplots ---

    # Define width ratios based on the number of countries in each plot for a balanced look
    num_countries_1 = len(order_1)
    num_countries_2 = len(order_2)

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(32, 10),  # A wide figure to accommodate both charts
        gridspec_kw={'width_ratios': [num_countries_1, num_countries_2]}  # Allocate space based on content
    )

    # --- Plot Chart 1 on the first axis (ax1) ---
    plot_chart_on_ax(
        ax=ax1,
        df=df_main,
        target_countries=target_1,
        ordered_countries=order_1,
        color_palette=map_script_colors,
        chart_title="Key Regions"
    )

    # --- Plot Chart 2 on the second axis (ax2) ---
    plot_chart_on_ax(
        ax=ax2,
        df=df_main,
        target_countries=target_2,
        ordered_countries=order_2,
        color_palette=map_script_colors,
        chart_title="Selected Developed Nations"
    )

    # --- Add a single, shared legend to the figure ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, [f'Cluster {label}' for label in labels],
               title='Pollution Cluster',
               loc='upper right',  # Position legend on the figure level
               bbox_to_anchor=(0.98, 0.95),  # Fine-tune position
               prop={'family': 'Arial', 'size': 12})
    plt.setp(fig.legends[0].get_title(), fontname='Arial', fontsize=14)

    # --- Final Adjustments and Saving ---
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for the shared legend

    try:
        plt.savefig(output_filename, format='pdf')
        print(f"✅ Success! Combined chart saved as '{output_filename}'")
    except Exception as e:
        print(f"❌ Error saving '{output_filename}': {e}")

    plt.close(fig)


# --- Execute the Main Function ---
if __name__ == '__main__':
    main()