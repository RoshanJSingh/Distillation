# Distillation

This repo shows a simple teacher student distillation project.

It uses Fashion-MNIST.
It is easy to run on a normal laptop.

The teacher is a small CNN in PyTorch.
The student is a manual multiclass logistic regression model.

The project also shows:

- semantic communication
- response based distillation
- heterogeneous models
- k-means++ coreset selection
- uncertainty based unlabeled sample picking
- pseudo labels from teacher to student
- manual gradient updates from prediction gap

## What is happening here

The idea is simple.
First we train a better teacher.
Then we train a weak student on small labeled data.
Then we try to improve the student with teacher outputs.

The student learns from two things:

- hard labels
- soft teacher probabilities

This is response based distillation.

## Semantic communication

In this project semantic communication means the teacher sends useful task meaning, not raw image pixels.

There are two message modes:

- `full` for the full probability vector
- `topk` for top class ids and top probabilities only

This is a compact message. It still keeps the class meaning.

## Why the models are heterogeneous

The models are not the same type.

- teacher: neural CNN
- student: linear softmax regression

So this is a heterogeneous teacher student setup.

## What is a coreset

A coreset is a small but useful subset from a bigger pool.

Here we do not use all unlabeled images in the same way.
We choose a smaller set that is useful and diverse.

We use teacher embeddings for this.
We cluster them with K-means++.
Then we pick strong points from different clusters.

## Why K-means++

K-means++ gives better starting centers than plain random start.
That usually gives cleaner clusters.
That helps the coreset stay more diverse.

## What uncertainty means

We score unlabeled samples with:

- teacher entropy
- teacher student disagreement

If entropy is high, the teacher is less sure.
If disagreement is high, teacher and student think differently.
Both cases are useful for selection.

## What label transfer means

The teacher gives targets for unlabeled data.

It gives:

- soft labels for distillation
- hard pseudo labels if confidence is above a threshold

This is how label transfer works here.

## What prediction divergence means

Prediction divergence means teacher and student give different class distributions on the same image.

The student is updated to reduce this difference.
The soft part of the loss uses KL divergence with temperature scaling.

## Manual gradient update

The student is not trained with `sklearn.LogisticRegression`.
The main student loop does manual updates.

For softmax regression the logit gradient follows:

```text
dL/dz = P - Q
```

`P` is the student distribution.
`Q` is the target distribution.

This target can be:

- a hard one hot label
- a soft teacher label
- a pseudo label from the teacher

## Project layout

```text
README.md
requirements.txt
configs/default.yaml
src/data.py
src/teacher_model.py
src/student_manual.py
src/semantic_message.py
src/coreset.py
src/distillation.py
src/metrics.py
scripts/train_teacher.py
scripts/train_student_baseline.py
scripts/select_coreset.py
scripts/train_student_distilled.py
scripts/evaluate_all.py
outputs/
tests/
```

## Install

Run this from the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run steps

1. Train teacher

```bash
python scripts/train_teacher.py --config configs/default.yaml
```

2. Train plain student baseline

```bash
python scripts/train_student_baseline.py --config configs/default.yaml
```

3. Train student with distillation only

```bash
python scripts/train_student_distilled.py --config configs/default.yaml --mode distill_only
```

4. Select coreset from unlabeled pool

```bash
python scripts/select_coreset.py --config configs/default.yaml
```

5. Train student with distillation and coreset transfer

```bash
python scripts/train_student_distilled.py --config configs/default.yaml --mode coreset
```

6. Evaluate all four settings

```bash
python scripts/evaluate_all.py --config configs/default.yaml
```

7. Run tests

```bash
python -m pytest tests
```

## Example output shape

The numbers will change.
But the output will look like this:

```text
{
  "checkpoint": "outputs/artifacts/teacher/best_teacher.pt",
  "best_val_acc": <float>
}
```

```text
{
  "accuracy": <float>,
  "macro_f1": <float>,
  "confusion_matrix": [...],
  "per_class": [...]
}
```

## Files to read first

1. `configs/default.yaml`
2. `src/data.py`
3. `src/teacher_model.py`
4. `src/student_manual.py`
5. `src/distillation.py`
6. `src/coreset.py`
7. `scripts/train_student_distilled.py`

