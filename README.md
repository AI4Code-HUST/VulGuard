# Realistic-Eval-JIT-VP
 
## Data:

Please following the link below to download the data.

- https://figshare.com/s/bedbc45f494aed760e06

## Folder Structure
      
```
Realistic-Eval-JIT-VP
│── README.md
│── JIT-DP
│   │── README.md
│   │── defectguard
│   │   │── crawler
│   │   │── models
│   │── pyszz_v2
│   │── scripts
│── JIT-VP
│   │── CodeJIT
│   │   |── graph_builder
|   |   |   │── README.md
│   |   │── model
│   │   |   │── README.md
│   │── VCCFinder
│   │   │── README.md
```

### Descriptions

- **JIT-DP**:
  - **defectguard**: our tool for analyzing software defects.
    - **crawler**: contains code for mining data.
    - **models**: contains models of Just-In-Time defect prediction implementations.
  - **pyszz_v2**: contains code of various SZZ algorithm implementations.
  - **scripts**: contains example scripts.

- **JIT-VP**: 
  - **CodeJIT**: contains code for 
    - **graph_builder**: contains code for constructing code graphs for CodeJIT model.
    - **model**: contains model of CodeJIT implemeteation.
  - **VCCFinder**: contains code of VCCFinder implementation.

### Note
For detail instructions, check out the README.md in each folder.