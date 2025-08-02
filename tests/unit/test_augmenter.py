"""Tests for active-learning data augmentation."""

from training.augmenter import ActiveLearningAugmenter


def _recall(dataset, val_set):
    return sum(1 for s in val_set if s in dataset) / len(val_set)


def test_augmenter_improves_recall_and_respects_interval():
    samples = ["quick brown fox", "lazy dog"]
    losses = [0.1, 1.0]
    synonyms = {"lazy": ["sluggish"], "dog": ["canine"]}
    augmenter = ActiveLearningAugmenter(synonyms, k=1, percentile=95, interval=2)
    val_set = ["quick brown fox", "lazy dog", "sluggish canine"]

    # Epoch 1: no augmentation should happen.
    ds_epoch1 = augmenter.augment(samples, losses, epoch=1)
    assert ds_epoch1 == samples
    recall_before = _recall(ds_epoch1, val_set)

    # Epoch 2: augmentation should add a paraphrased variant and improve recall.
    ds_epoch2 = augmenter.augment(samples, losses, epoch=2)
    assert "sluggish canine" in ds_epoch2
    recall_after = _recall(ds_epoch2, val_set)

    assert recall_after - recall_before >= 0.01
