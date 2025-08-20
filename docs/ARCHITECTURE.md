# Architecture

## Project Structure

```
VulGuard/
├── dg_cache/               # Vulguard output
├── docs/                   # Documentation files
├── scripts/                # Utility scripts
├── vulguard/               # Main application source code
│   ├── crawler/            # Code mining and repository crawler modules
│   │   ├── features/       # Feature extraction logic
│   │   ├── graph_builder/  # Source code graph generation
│   │   ├── szz/            # SZZ algorithm
│   ├── models/             # JIT-VP model definitions
|   |   ├── init_model.py   # Models declaration
|   |   ├── BaseWraper.py   # Wraper template
│   ├── utils/              # Core utilities shared across VulGuard
```

## Component Descriptions

### `crawler/`
The `crawler` module is responsible for mining the source code repository, tracking changes, and building the raw data.

- **features/**:  
  Handles static and dynamic feature extraction from code changes.

- **graph_builder/**:  
  Constructs code graphs from code to enable graph-based learning and contextual vulnerability detection.

- **szz/**:  
  Implements the SZZ algorithm to trace back vulnerability-fixing commits to their vulnerability-inducing commits, essential for labeling data.

- **utils/**:  
  Helper functions and utilities for code parsing, file handling, etc.

### `models/`
This module contains JIT-VP model for vulnerability prediction.

- Supports multiple architectures such as:
  - Classical models (Random Forests, SVM)
  - Neural networks 
  - Graph Neural Networks 

- You can easily **add your own models** or customize existing ones.

### `docs/`
Includes user guides, usage examples, and internal documentation.

## Extending VulGuard

- **Add New Models**:  
  Add your model under `models/` by define your model wrapper following the template.

- **Customize Features**:  
  Extend the `features/` module to extract new metrics or representations tailored to your analysis goals.

