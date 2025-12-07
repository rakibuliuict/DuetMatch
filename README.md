# DuetMatch: Harmonizing Semi-Supervised Brain MRI Segmentation via Decoupled Branch Optimization

## Installation

Environment:
- Python 3.9

Install dependencies:

    pip install -r requirements.txt


## Data Format

Dataset structure:

    name/
        data/
            case001.h5
            case002.h5
            *.h5
        train.list
        val.list
        test.list


## Train

Run training:

    python train_duetmatch.py --exp exp_name --gpu 0 --labelnum 10


## Test

Run testing:

    python test_duetmatch.py --exp exp_name --gpu 0 --labelnum 10

---

## Acknowledge
[BCP](https://github.com/DeepMed-Lab-ECNU/BCP)


