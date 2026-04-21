from __future__ import annotations

import numpy as np

from src.student_manual import ManualSoftmaxRegression, softmax


def test_softmax_shape_and_normalization() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    probs = softmax(logits)
    assert probs.shape == logits.shape
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(logits.shape[0]), atol=1e-6)


def test_manual_gradient_shapes() -> None:
    model = ManualSoftmaxRegression(input_dim=4, num_classes=3, seed=0)
    X = np.random.randn(5, 4).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    soft_targets = np.full((5, 3), 1.0 / 3.0, dtype=np.float32)

    grads = model._loss_and_gradients(
        X,
        hard_labels=y,
        soft_targets=soft_targets,
        alpha=0.5,
        temperature=2.0,
    )

    assert grads["grad_W"].shape == model.W.shape
    assert grads["grad_b"].shape == model.b.shape


def test_manual_update_improves_tiny_synthetic_problem() -> None:
    X = np.array([[2.0, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]], dtype=np.float32)
    y = np.array([1, 1, 0, 0], dtype=np.int64)

    model = ManualSoftmaxRegression(input_dim=2, num_classes=2, seed=1, init_scale=0.01)
    initial_probs = model.predict_proba(X)
    initial_loss = -np.mean(np.log(initial_probs[np.arange(len(y)), y] + 1e-12))

    history = model.fit(
        X=X,
        hard_labels=y,
        X_val=X,
        y_val=y,
        epochs=50,
        batch_size=2,
        learning_rate=0.2,
        seed=1,
    )
    final_probs = model.predict_proba(X)
    final_loss = -np.mean(np.log(final_probs[np.arange(len(y)), y] + 1e-12))

    assert history.train_loss[-1] < initial_loss
    assert final_loss < initial_loss
