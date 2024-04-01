[🇷🇺 Русская версия](README_ru.md)

# Diploma Project: Scientific Text Vectorization
![MindMap](/img/markmap-main.png)

This repository contains the code for the diploma project focused on vectorizing scientific texts using the Top2Vec algorithm.

## Overview

The aim of the project is to develop a module for analyzing scientific texts, identifying development trends, searching for similar documents, and determining the pace of development of thematic groups. The practical application of the module is intended for the analysis of individual thematic groups in the field of computer science.

## Key Features

- Automated data collection from the arXiv platform.
- Vectorization of scientific texts using the Top2Vec algorithm.
- Identification of key words for thematic groups.
- Clustering and analysis of thematic groups.
- Trend analysis and identification of key development directions.
- Visualization of analysis results.

## Dataset

The model was trained on a dataset of scientific paper abstracts obtained from [arXiv](https://arxiv.org/). The dataset covers a range of topics in the field of computer science from 2010 to 2024.

You can access the dataset [arxiv_papers_cs](https://huggingface.co/datasets/CCRss/arxiv_papers_cs).

## Usage

The main project code is contained in the `main.ipynb` Jupyter notebook. To work with the notebook:

1. Open the notebook in Jupyter Lab or Jupyter Notebook.
2. Execute the cells in order, to start the different stages of the analysis.

## Examples

Here are some examples of the model's output for the thematic group "UAVs in Natural Disasters and Emergencies":

### Trend Analysis for "UAVs in Natural Disasters and Emergencies"

![Trend Analysis](/img/disasters_and_emergency_plot.png)

This graph shows the trend of interest in the use of UAVs in situations of natural disasters and emergencies over time.

### Key Metrics Table

Analysis for the Thematic Group: Natural Disasters and Emergencies
|   Year |   Number of Publications |   Growth Acceleration |   Change in Number of Publications | Relative Growth   |
|-------:|-------------------------:|----------------------:|-----------------------------------:|:------------------|
|   2010 |                       19 |                     0 |                                  0 | 0.0%              |
|   2011 |                       15 |                    -4 |                                 -4 | -21.05%           |
|   2012 |                       28 |                    17 |                                 13 | 86.67%            |
|   2013 |                       38 |                    -3 |                                 10 | 35.71%            |
|   2014 |                       28 |                   -20 |                                -10 | -26.32%           |
|   2015 |                       47 |                    29 |                                 19 | 67.86%            |
|   2016 |                       63 |                    -3 |                                 16 | 34.04%            |
|   2017 |                       94 |                    15 |                                 31 | 49.21%            |
|   2018 |                      173 |                    48 |                                 79 | 84.04%            |
|   2019 |                      266 |                    14 |                                 93 | 53.76%            |
|   2020 |                      337 |                   -22 |                                 71 | 26.69%            |
|   2021 |                      380 |                   -28 |                                 43 | 12.76%            |
|   2022 |                      453 |                    30 |                                 73 | 19.21%            |
|   2023 |                      509 |                   -17 |                                 56 | 12.36%            |

## Contribution

We welcome contributions to the top2vec_scientific_texts model. If you have suggestions, improvements, or encounter any issues, please feel free to open an issue or submit a pull request.

## Results

The results of the analysis include the identification of thematic groups, trend analysis, and visualization of the dynamics of interest in various topics over the years. These results are presented in the form of tables and graphs inside the `main.ipynb` notebook.
