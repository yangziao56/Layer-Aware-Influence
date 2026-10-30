# Layer-Aware Influence

**Online data valuation, viewed layer by layer.**

[PDF](https://arxiv.org/pdf/2510.16007) · [Paper](https://arxiv.org/abs/2510.16007) · [Experiment directories](#choose-an-experiment)

![Layer-aware influence: a layered view of training-data value.](assets/readme-banner.svg)

*Layer-Aware Influence for Online Data Valuation Estimation*  
**Ziao Yang**, Longbo Huang, and Hongfu Liu

## Choose an experiment

| **Language-model training** | **Task evaluation** |
| :--- | :--- |
| [`GPT-Neo/`](GPT-Neo/) — training entry point and custom trainer | [`GLUE/`](GLUE/) — GLUE experiments and result summaries |
| **TPU workflows** | **Text datasets** |
| [`DBC_tpu/`](DBC_tpu/) — TPU training and data curation | [`Emotion_20News/`](Emotion_20News/) — Emotion and 20 Newsgroups |

### Follow the implementation

[`gpt-neo.py`](GPT-Neo/gpt-neo.py) → [`ghost_trainer.py`](GPT-Neo/ghost_trainer.py)  
[`evaluation_single_task_ours.py`](GLUE/evaluation_single_task_ours.py) → [`my_trainer.py`](GLUE/my_trainer.py)

```bash
git clone https://github.com/yangziao56/Layer-Aware-Influence.git
cd Layer-Aware-Influence
```

Choose an experiment directory and configure its model, data, and output paths for your environment. The repository includes GPU and TPU implementations.

<details>
<summary><strong>Citation</strong></summary>

```bibtex
@misc{yang2025layer,
  title = {Layer-Aware Influence for Online Data Valuation Estimation},
  author = {Yang, Ziao and Huang, Longbo and Liu, Hongfu},
  year = {2025},
  eprint = {2510.16007},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2510.16007}
}
```

</details>
