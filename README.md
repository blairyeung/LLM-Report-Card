# Skill Reports: Qualitative Evaluation of Language Models Using Natural Language Summaries

This repository contains the code and data for the paper "Skill Reports: Qualitative Evaluation of Language Models Using Natural Language Summaries".

If you find this helpful, please reference in your paper:
```bibtex
@inproceedings{yang2024skillreports,
  title={Skill Reports: Qualitative Evaluation of Language Models Using Natural Language Summaries},
  author={Blair Yang, Fuyang Cui, Keiran Paster, Jimmy Ba, Passhootan Vaezipoor, Silviu Pitis, Michael R. Zhang},
  year={2024}
}
```

To learn more,
- [Research Paper]()
- [Project Website](https://sites.google.com/view/llm-skill-reports/home)

# Instructions

## Environment Setup

```bash
conda create -n skillreports python=3.11
conda activate skillreports
pip install -r requirements.txt
```

## Repository Structure

```
- cards/ - contains many skill reports
- code/
    - card_gen/ - code for generating skill reports
    - core/ - core utilities
    - dataset/ - dataset related utilities
    - eval/ - different skill report evaluation metrics
    - ... - other utilities
- data/ - formatted datasets
- prompts/ - prompts for generating and evaluating skill reports
```

Many of the code files contain example use cases.

If you have any question, please feel free to open an issue or contact the authors.
