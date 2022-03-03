# Collecting Geospatial Data under Local Differential Privacy with Improving Frequency Estimation

## Prerequisites
- Anaconda Python 3.8
- MATLAB R2020b
  - Install MATLAB Engine API for Python with referring to
  https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
- Clone the project
  ```
  git clone https://github.com/daeyounghong/square-mechanism
  ```

## Configuration
Create `config.json` in the project directory and write the contents as follows:
```
{
  "data_dir": "data",
  "eps": 12.5,
  "grid_shape": [600, 300]
}
```
- `"data_dir"`: the path of a directory where dataset file and output files are stored.
- `"eps"`: the privacy budget used for running the algorithm.
- `"grid_shape"`: the grid shape used in the proposed postprocessing for frequency estimation based on convex optimization.

## Preprocessing
- Download [Gowalla dataset](https://snap.stanford.edu/data/loc-gowalla.html)
- Place Gowalla data file `loc-gowalla_totalCheckins.txt` on `data_dir` of `config.json`
- Run the following command:
  ```
  python prepro.py
  ```
- The preprocessed file is stored in `data_dir` of `config.json`

## Running the Algorithm
- Run the following command:
  ```
  python run_alg.py
  ```
