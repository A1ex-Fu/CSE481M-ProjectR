# SciLead: Scientific Leaderboard Dataset

Name: SciLead: Scientific Leaderboard Dataset

Version: 0.1

Authors: Furkan Şahinuç (UKP Lab, Technical University of Darmstadt), Thy Thy Tran (UKP Lab, Technical University of Darmstadt), Yulia Grishina (Amazon Alexa AI - Berlin), Yufang Hou (IBM Research Europe - Ireland), Bei Chen (Amazon Alexa AI - Berlin), Iryna Gurevych (UKP Lab, Technical University of Darmstadt)

The components of this dataset are used in the experiments of the paper "[Efficient Performance Tracking: Leveraging Large Language Models for Automated Construction of Scientific Leaderboards](https://arxiv.org/abs/2409.12656)" at EMNLP 2024 main conference. For further information please refer to paper's [GitHub repository](https://github.com/UKPLab/leaderboard-generation).

If you utilize this repository and our work, please cite our paper.

```bibtex
@article{sahinuc2024efficient,
  title     = {Efficient Performance Tracking: Leveraging Large Language Models for Automated Construction of Scientific Leaderboards},
  author    = {{\c{S}}ahinu{\c{c}}, Furkan and Tran, Thy Thy and Grishina, Yulia and Hou, Yufang and Chen, Bei and Gurevych, Iryna},
  journal   = {arXiv preprint arXiv:2409.12656},
  year      = {2024},
  url       = {https://arxiv.org/abs/2409.12656}
}
```

✉️ Contact person: Furkan Şahinuç, [furkan.sahinuc@tu-darmstadt.de](mailto:furkan.sahinuc@tu-darmstadt.de)

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Introduction

The dataset consist of Task, Dataset, Metric and Result (TDMR) information from scientific papers along with a leaderboard dataset constructed based on TDMR information. Content information is as follows:

* ```tdm_annotations.tsv```: TSV file containing Task, Dataset, Metric and Result values from selected scientific papers. This file also contains URLs of papers. 


* ```tdm_annotations.json```: JSON version of ```tdm_annotations.tsv```.


* ```leaderboards.json```: JSON file containing leaderboards constructed by grouping the papers with the same Task, Dataset, Metric tuples. The corresponding result values of the grouped papers are added to the leaderboards. Threshold for minimum number of values for a leaderboard is 3. Therefore, not every data e instance of ```tdm_annotations.tsv``` has to belong to a leaderboard. Similarly, a paper can belong to different leaderboards with different results obtained for different tasks.

### Column Description

Descriptions of the columns are given below:

| Column Name     | Description                                                               |
|-----------------|---------------------------------------------------------------------------|
| ```PaperURL```  | URL of the paper                                                          |
| ```PaperName``` | Identifier name for the paper                                             |
| ```Task```      | Studied task name in the paper                                            |
| ```Dataset```   | Studied dataset name in the paper                                         |
| ```Metric```    | Used evaluation metric  in the paper                                      |
| ```Result```    | Result value obtained for corresponding Task, Dataset, Metric (TDM) tuple |
| ```Comment```   | Comment about data instances if necessary                                 |

### Data Loading

```python
import pandas as pd
tdmr_data = pd.read_csv("tdm_annotations.tsv", sep="\t")
```
or

```python
import json
tdmr_data = json.load(open("tdm_annotations.json", 'r'))
```

## Dataset Statistics

| Dataset item              | Counts |
|---------------------------|--------|
| Papers                    | 43     |
| Leaderboards              | 27     |
| Avg. papers / leaderboard | 5.19   |
| TDMRs                     | 295    |
| Unique tasks              | 23     |
| Unique datasets           | 71     |
| Unique metrics            | 26     |
| Unique TDMs               | 138    |

## Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
