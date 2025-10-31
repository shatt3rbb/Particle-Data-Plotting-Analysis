# Analysis Project

## Overview
This project is designed for analyzing particle physics data. It includes functionality for loading data from ROOT files, processing the data, applying analysis cuts, and generating plots and yield tables. The main analysis logic is implemented in the `src/stepRun.py` file.

## Directory Structure
```
analysis-project
├── src
│   ├── stepRun.py          # Main analysis script
│   └── config
│       └── config.yaml     # Configuration file for sample paths, variable binning, and scaling factors
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Setup
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd analysis-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the sample paths, variable binning, and scaling factors in `src/config/config.yaml` as needed.

## Usage
To run the analysis, execute the following command:
```
python src/stepRun.py <region> <control_type> <event_type> <scaling_option>
Basic Example:
python src/stepRun.py SR SR ee Unscaled
```
Replace `<region>`, `<control_type>`, `<event_type>`, and `<scaling_option>` with the appropriate values based on your analysis requirements.

## Example
An example command to run the analysis might look like this:
```
python src/stepRun.py SR 3lCR all Scaled_CR1_NB
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.