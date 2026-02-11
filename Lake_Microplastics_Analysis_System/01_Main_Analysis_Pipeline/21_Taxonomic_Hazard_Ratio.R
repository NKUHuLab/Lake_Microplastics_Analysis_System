# =================================================================
# Final Polished Plot (Version 3.1 - Corrected)
# - Fixed error for missing font size variable
# - Increased size of all text and plot elements for clarity
# - Optimized vertical spacing for a more compact layout
# =================================================================

# --- 0. Load Libraries ---
library(tidyverse)
library(patchwork)
library(extrafont)
library(Cairo)


# --- 1. Visual Parameters (Easy to Adjust Here) ---
# Font Sizes
FONT_SIZE_BASE <- 14       # Base size for axis titles, legends
FONT_SIZE_SPECIES <- 12    # Y-axis species names
FONT_SIZE_FAMILY <- 4.5      # Family names (geom_text uses a different scale)
FONT_SIZE_CLASS <- 14      # Facet labels (Class names)
FONT_SIZE_TITLE <- 18      # Titles for plots A and B
FONT_SIZE_SUPERTITLE <- 22 # The very top title

# Point and Error Bar Sizes
POINT_SIZE <- 5.0          # Size of the circular points
POINT_STROKE <- 0.7        # Outline thickness of the points
ERRORBAR_HEIGHT <- 0.6     # Height of the error bar end caps
ERRORBAR_LINEWIDTH <- 0.9  # Thickness of the error bar line

# Spacing
GAP_FAMILY <- 1.5          # Reduced gap, but still distinct
STEP_SPECIES <- 0.9        # Reduced space for each species to make plot less long


# --- 2. Load Data ---
all_species_path <- "E:/lake-MP-W/dataset/FAO/analysis_results_all_species_with_taxa.csv"
all_species_data <- read.csv(all_species_path)


# --- 3. Reclassify Single-Species Families in 'Osteichthyes' ---
single_species_families <- all_species_data %>%
  filter(class == "Osteichthyes") %>%
  group_by(family) %>%
  filter(n() == 1) %>%
  pull(family)

all_species_data <- all_species_data %>%
  mutate(family = if_else(
    class == "Osteichthyes" & family %in% single_species_families,
    "Other Osteichthyes",
    family
  ))


# --- 4. Advanced Data Prep for Hybrid Spacing ---
plot_data <- all_species_data %>%
  mutate(
    aqua_conf_low = aquaculture_coef - 1.96 * aquaculture_stderr,
    aqua_conf_high = aquaculture_coef + 1.96 * aquaculture_stderr,
    capture_conf_low = capture_coef - 1.96 * capture_stderr,
    capture_conf_high = capture_coef + 1.96 * capture_stderr
  ) %>%
  arrange(class, family, aquaculture_coef) %>%
  group_by(class) %>%
  mutate(
    is_new_family = (family != lag(family, default = first(family))),
    gap = ifelse(is_new_family, STEP_SPECIES + GAP_FAMILY, STEP_SPECIES),
    y_pos = rev(cumsum(gap))
  ) %>%
  ungroup()

family_labels <- plot_data %>%
  group_by(class, family) %>%
  summarise(y_label_pos = mean(y_pos), .groups = 'drop')


# --- 5. Create the Bolder Plots ---

# Plot A: Aquaculture Forest Plot
forest_plot_aqua <- ggplot(plot_data, aes(x = aquaculture_coef, y = y_pos)) +
  geom_errorbarh(aes(xmin = aqua_conf_low, xmax = aqua_conf_high),
                 height = ERRORBAR_HEIGHT, linewidth = ERRORBAR_LINEWIDTH, color = "grey20") +
  geom_vline(xintercept = 1, linetype = "solid", color = "black", linewidth=0.6) +
  geom_point(aes(fill = aquaculture_pvalue < 0.05),
             shape = 21, size = POINT_SIZE, stroke = POINT_STROKE, color = "black") +
  geom_text(data = family_labels, aes(label = family, y = y_label_pos),
            x = -Inf, hjust = -0.1, size = FONT_SIZE_FAMILY, fontface = "italic", color = "gray10") +
  facet_grid(class ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(values = c("TRUE" = "#c44e52", "FALSE" = "white"), name = "Significant (p < 0.05)") +
  scale_y_continuous(breaks = plot_data$y_pos, labels = plot_data$species) +
  labs(
    title = "A: Aquaculture 'Plastic Footprint'",
    x = "Hazard Ratio (95% CI)",
    y = ""
  ) +
  theme_bw(base_family = "Arial", base_size = FONT_SIZE_BASE) +
  theme(
    # CORRECTED: Used FONT_SIZE_TITLE which is correctly defined above
    plot.title = element_text(size = FONT_SIZE_TITLE, face = "bold", hjust=0.5),
    legend.position = "bottom",
    strip.text.y.left = element_text(angle = 0, face = "bold", size = FONT_SIZE_CLASS),
    strip.background = element_rect(fill = "grey90", color = "grey90"),
    strip.placement = "outside",
    axis.text.y = element_text(size = FONT_SIZE_SPECIES), # Control species font size
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(color="black"),
    panel.spacing.y = unit(1.0, "lines")
  )

# Plot B: Capture Fisheries Forest Plot
forest_plot_capture <- ggplot(plot_data, aes(x = capture_coef, y = y_pos)) +
  geom_errorbarh(aes(xmin = capture_conf_low, xmax = capture_conf_high),
                 height = ERRORBAR_HEIGHT, linewidth = ERRORBAR_LINEWIDTH, color = "grey20") +
  geom_vline(xintercept = 1, linetype = "solid", color = "black", linewidth=0.6) +
  geom_point(aes(fill = capture_pvalue < 0.05),
             shape = 21, size = POINT_SIZE, stroke = POINT_STROKE, color="black") +
  facet_grid(class ~ ., scales = "free_y", space = "free_y") +
  scale_fill_manual(values = c("TRUE" = "#4c6a9c", "FALSE" = "white"), name = "Significant (p < 0.05)") +
  scale_y_continuous(breaks = plot_data$y_pos, labels = plot_data$species) +
  labs(
    title = "B: Capture 'Plastic Footprint'",
    x = "Hazard Ratio (95% CI)",
    y = ""
  ) +
  theme_bw(base_family = "Arial", base_size = FONT_SIZE_BASE) +
  theme(
    # CORRECTED: Used FONT_SIZE_TITLE which is correctly defined above
    plot.title = element_text(size = FONT_SIZE_TITLE, face = "bold", hjust=0.5),
    legend.position = "bottom",
    strip.text.y = element_blank(),
    strip.background = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line.x = element_line(color="black"),
    panel.spacing.y = unit(1.0, "lines")
  )

# --- 6. Combine Plots and Save ---
final_plot <- forest_plot_aqua + forest_plot_capture +
  plot_annotation(
    title = "Species-Specific Comparison of Fishery Impacts by Taxonomic Group",
    theme = theme(plot.title = element_text(family = "Arial", size = FONT_SIZE_SUPERTITLE, face = "bold", hjust = 0.5))
  )

# Print the final plot to the RStudio viewer
print(final_plot)

# Save the final plot as a high-quality PDF
output_pdf_path <- "E:/lake-MP-W/draw/FAO-MP/SQB_Forest_Plots_Bolder_Final.pdf"
ggsave(
  output_pdf_path,
  plot = final_plot,
  device = cairo_pdf,
  width = 18,
  height = 0.35 * nrow(plot_data) + 2.5,
  limitsize = FALSE
)

print(paste0("\nCorrected plot saved to:\n", output_pdf_path))