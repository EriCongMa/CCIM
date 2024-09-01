# CCIM


CCIM: Cross-Modal Cross-Lingual Interactive Image Translation.

The official repository for EMNLP 2023 Findings conference paper: 

- **Cong Ma**, Yaping Zhang, Mei Tu, Yang Zhao, Yu Zhou, Chengqing Zong. **CCIM: Cross-Modal Cross-Lingual Interactive Image Translation**. In Findings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), Singapore. December 6-10, 2023. pp. 4959–4965. [ACL_Anthology](https://aclanthology.org/2023.findings-emnlp.330/)



## 1. Introduction

Text image machine translation (TIMT) which translates source language text images into target language texts has attracted intensive attention in recent years. Although the end- to-end TIMT model directly generates target translation from encoded text image features with an efficient architecture, it lacks the recognized source language information resulting in a decrease in translation performance. In this paper, we propose a novel Cross-modal Cross-lingual Interactive Model (CCIM) to incorporate source language information by synchronously generating source language and target language results through an interactive attention mechanism between two language decoders. Extensive experimental results have shown the interactive decoder significantly outperforms end-to-end TIMT models and has faster decoding speed with smaller model size than cascade models.



<img src="./Figures/model.pdf" style="zoom:60%;" />



## 2. Usage

### 2.1 Requirements

- python==3.6.2
- pytorch == 1.3.1
- torchvision==0.4.2
- numpy==1.19.1
- lmdb==0.99
- PIL==7.2.0
- jieba==0.42.1
- nltk==3.5
- six==1.15.0
- natsort==7.0.1



### 2.2 Train the Model

```shell
bash ./train_model_guide.sh
```



### 2.3 Evaluate the Model

```shell
bash ./test_model_guide.sh
```



### 2.4 Datasets

We use the dataset released in [E2E_TIT_With_MT](https://github.com/EriCongMa/E2E_TIT_With_MT/tree/main).



## 3. Acknowledgement

The reference code of the provided methods are:

- [EriCongMa](https://github.com/EriCongMa)/[**E2E_TIT_With_MT**](https://github.com/EriCongMa/E2E_TIT_With_MT)
- [clovaai](https://github.com/clovaai)/**[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)**
- [OpenNMT](https://github.com/OpenNMT)/**[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)**
- [THUNLP-MT](https://github.com/THUNLP-MT)/**[THUMT](https://github.com/THUNLP-MT/THUMT)**


We thanks for all these researchers who have made their codes publicly available.



## 4. Citation

If you want to cite our paper, please use this bibtex version:

- ACL Anthology offered bib citation format

  - ```latex
    @inproceedings{ma-etal-2023-ccim,
        title = "{CCIM}: Cross-modal Cross-lingual Interactive Image Translation",
        author = "Ma, Cong  and
          Zhang, Yaping  and
          Tu, Mei  and
          Zhao, Yang  and
          Zhou, Yu  and
          Zong, Chengqing",
        editor = "Bouamor, Houda  and
          Pino, Juan  and
          Bali, Kalika",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
        month = dec,
        year = "2023",
        address = "Singapore",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.findings-emnlp.330",
        doi = "10.18653/v1/2023.findings-emnlp.330",
        pages = "4959--4965",
        abstract = "Text image machine translation (TIMT) which translates source language text images into target language texts has attracted intensive attention in recent years. Although the end-to-end TIMT model directly generates target translation from encoded text image features with an efficient architecture, it lacks the recognized source language information resulting in a decrease in translation performance. In this paper, we propose a novel Cross-modal Cross-lingual Interactive Model (CCIM) to incorporate source language information by synchronously generating source language and target language results through an interactive attention mechanism between two language decoders. Extensive experimental results have shown the interactive decoder significantly outperforms end-to-end TIMT models and has faster decoding speed with smaller model size than cascade models.",
    }
    ```

- Semantic Scholar offered bib citation format

  - ```latex
    @inproceedings{Ma2023CCIMCC,
      title={CCIM: Cross-modal Cross-lingual Interactive Image Translation},
      author={Cong Ma and Yaping Zhang and Mei Tu and Yang Zhao and Yu Zhou and Chengqing Zong},
      booktitle={Conference on Empirical Methods in Natural Language Processing},
      year={2023},
      url={https://api.semanticscholar.org/CorpusID:266176714}
    }
    ```

- DBLP offered bib citation format

  - ```latex
    @inproceedings{DBLP:conf/emnlp/MaZTZZZ23,
      author       = {Cong Ma and
                      Yaping Zhang and
                      Mei Tu and
                      Yang Zhao and
                      Yu Zhou and
                      Chengqing Zong},
      editor       = {Houda Bouamor and
                      Juan Pino and
                      Kalika Bali},
      title        = {{CCIM:} Cross-modal Cross-lingual Interactive Image Translation},
      booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                      2023, Singapore, December 6-10, 2023},
      pages        = {4959--4965},
      publisher    = {Association for Computational Linguistics},
      year         = {2023},
      url          = {https://doi.org/10.18653/v1/2023.findings-emnlp.330},
      doi          = {10.18653/V1/2023.FINDINGS-EMNLP.330},
      timestamp    = {Fri, 16 Aug 2024 07:47:11 +0200},
      biburl       = {https://dblp.org/rec/conf/emnlp/MaZTZZZ23.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
    ```



If you have any issues, please contact with [email](macong275262544@outlook.com).
