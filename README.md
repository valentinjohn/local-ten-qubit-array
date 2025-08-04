# Data Repository for Scientific Publication

The repository is organized to facilitate reproducibility of the results presented in the paper and includes raw data, processed data, and scripts for generating the processed data and all figures.

## Repository Structure

### Raw Data
All raw data used in the publication is stored in `./code_repo/data/uuid_datasets`. These files contain the unprocessed datasets referenced in the publication.

### Extracted Data
Processed data is located directly within the `./code_repo/data/` directory. These files are derived from the raw data and are ready for use in analysis and figure generation.

### Reproducing Figures
- **Main Figures**: Use the Jupyter Notebook `final_figures.ipynb` to reproduce the main figures presented in the publication.
- **Supplemental Figures**: Scripts for generating supplemental figures are located in the corresponding folders

### Data Extraction
The scripts for extracting data from the raw datasets can be found in the corresponding folders. Follow the instructions below to reproduce the data extraction process.

## Prerequisites

1. **Python Dependencies**:
   - Install the required Python packages listed in `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

2. **core_tools Package (only needed for data extraction)**:
   - Clone the `core_tools` package from its repository:
     ```
     git clone https://github.com/stephanlphilips/core_tools.git
     ```
   - Follow the installation instructions in the `core_tools` repository.

## Usage Instructions

### Reproducing Data Extraction
1. Ensure that the `core_tools` package is installed and accessible in your Python environment.
2. Navigate to the `data_extraction_code/` folder and execute the scripts in the required sequence to reproduce the extracted data.

### Generating Figures
1. Open the `final_figures.ipynb` notebook in a Jupyter environment, and execute the cells in sequence to generate the main figures.
2. For supplemental figures, refer to the scripts in the corresponding folder.
