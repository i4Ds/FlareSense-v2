# Copilot Instructions for FlareSense v2

## Guidelines for a Good Python Software Engineer

- **Code Style**: Follow PEP 8 conventions for readable, consistent code. Use tools like Black or flake8.
- **Type Hints**: Add type annotations to functions and variables for better clarity and IDE support.
- **Modularity**: Write reusable, modular functions and classes. Avoid monolithic code.
- **Error Handling**: Use try-except blocks appropriately; log errors with the logging module.
- **Testing**: Write unit tests with pytest; aim for high coverage.
- **Performance**: Profile code with cProfile; optimize bottlenecks.
- **Documentation**: Use docstrings for functions; keep code self-documenting.
- **Version Control**: Commit frequently with clear messages; use branches for features.
- **Security**: Validate inputs; avoid common vulnerabilities like SQL injection or XSS in web apps.
- **Best Practices**: Use virtual environments; keep dependencies in requirements.txt; review code.

## Overall Project Description

FlareSense v2 is an experimental project focused on detecting solar radio bursts using machine learning. It trains a convolutional neural network (CNN) on e-Callisto radio spectrograms, which are visual representations of radio frequency data from solar observations. The repository includes:

- Training scripts (e.g., main.py) driven by YAML configs.
- Prediction utilities for datasets and live inference via Gradio interface (pred_live.py).
- Data processing modules (ecallisto_dataset.py, ecallisto_model.py).
- Jupyter notebooks for EDA, evaluation, and visualization in _notebooks/.
- Web app (app.py) for user interaction, likely built with Flask or similar.

The goal is to enhance app.py to improve the website's usability, design, and functionality, such as better UI/UX, responsive layout, and integration with prediction models.

## Website Overview and Folder Structure

The web app is primarily in `app.py`, providing a user interface for interacting with the prediction models. It likely includes features for uploading data, running predictions, and displaying results.

Key folder structure:

- **Root level**: Core scripts (`app.py`, `main.py`, `pred_live.py`, `pred_dataset.py`), data files (`ecallisto_radio_sunburst_train.csv`, `ecallisto_radio_sunburst_test.csv`), and config files (`requirements.txt`, `README.md`).
- **`_notebooks/`**: Jupyter notebooks for exploratory data analysis, evaluation, and visualization (e.g., `eval.ipynb`, `gradcam.ipynb`).
- **`artifacts/`**: Saved model checkpoints and artifacts (e.g., `best_model:v130/`).
- **`configs/`**: YAML configuration files for training and prediction (e.g., `best_v2.yml`).
- **`images/`**: Static images used in the app or for documentation.
- **`style/`**: Styling files, such as CSS or images for the web interface.
- **`tmp/`**: Temporary data directories (e.g., `Australia-ASSA_56/` for processed data).
- **`wandb/`**: Weights & Biases logs for experiment tracking.
- **`XAI/`**: Additional data, plots, and exports (e.g., precision-recall plots, heatmaps).
- **`burst_plots/`**: Generated plots of detected bursts.
- **`cleaned_data_pred/`**: Processed prediction data files.

## Environment

Use the conda environment: `conda activate flaresense-v2`

## Current Update Task for app.py

Update the website to group bursts by time (15-minute intervals) and display them as unified events. Key features:

- **Grouping**: Bursts within the same 15-minute window are grouped together.
- **Sorting**: Display newest groups first (today, then yesterday, etc.).
- **Display**: For each group, show a short title (e.g., "Burst at [time]"), then the burst images from different antennas side by side.
- **Within Group Sorting**: Sort images within each group by confidence (highest first).
- **UI Enhancements**: Add a simple toolbar/sidebar on the left for additional info (e.g., station details, links).
- **Simplicity**: Keep the interface clean, no unnecessary explanations.
- **Data Handling**: Use existing data loading functions, modify to group by rounded datetime (e.g., floor to 15 minutes).
- **Imports**: Ensure necessary imports are added (e.g., from datetime import timedelta if needed; already have datetime, pandas, etc.).

Continuously update this section with new imports, changes, or progress to maintain context for future sessions.

### Recent Changes:
- Added `TimeGroup` column in `load_image_paths` using `df['TimeGroup'] = df['Datetime'].dt.floor('15min')`.
- Modified `load_images_and_table` to group by `TimeGroup`, sort groups by newest first, and within groups by confidence descending.
- Updated gallery labels to include "Burst at [time] - Confidence: [value] %" for each image.
- Added a sidebar in `create_demo` with links to E-Callisto stations, contact, paper, and update info.
- Ensured gallery displays images in order, grouping them visually by time.
- **Major UI Overhaul**: Completely redesigned the interface with main content showing latest bursts and sidebar with tools.
- **New `get_latest_bursts` function**: Creates HTML display showing bursts grouped by time with instrument names and images.
- **Moved advanced features**: Date selection, gallery, table, and plots moved to collapsible accordions in right sidebar.
- **Clean main view**: Now shows burst titles with instrument detections in a card-like layout.
- **Updated styling**: Added emojis, better spacing, and modern card design for burst display.
- **Fixed image display**: Corrected image path format from `/file=` to `file/` for proper Gradio serving.
- **Grouped burst functionality**: Added shared `get_grouped_bursts_display` function used by both latest bursts and advanced tools.
- **ZIP export feature**: Added `create_zip_export` function that creates ZIP files with CSV and all burst images.
- **Simplified data browser**: Replaced complex interactive features with simple database overview and statistics.
- **Advanced tools enhancement**: Added grouped burst display and minimum stations filter to advanced search.
- **Station filtering**: Latest bursts now require detection by at least 3 stations (user-configurable) to reduce false positives.
- Added imports: `zipfile`, `shutil` for ZIP export functionality.
