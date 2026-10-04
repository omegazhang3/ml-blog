---
title: "Linear Regression from Scratch"
date: 2026-10-04
draft: false
math: true
tags: ["regression", "fundamentals", "gradient-descent"]
summary: "Deriving linear regression and implementing gradient descent in NumPy."
---

My first note: linear regression. This post is here to confirm that **math** and
**code highlighting** both render correctly.

## The model

We fit a linear model $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ by minimizing the
mean squared error:

$$
J(\mathbf{w}, b) = \frac{1}{2n} \sum_{i=1}^{n} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

The gradient with respect to the weights is:

$$
\nabla_{\mathbf{w}} J = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{y}^{(i)} - y^{(i)} \right) \mathbf{x}^{(i)}
$$

Inline math works too: the learning rate $\alpha$ controls the step size.

## Implementation

```python
import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        y_hat = X @ w + b
        error = y_hat - y
        grad_w = (X.T @ error) / n
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
```

## Takeaways

- Gradient descent moves parameters opposite the gradient.
- A learning rate that's too large diverges; too small is slow.

> Next up: logistic regression and the cross-entropy loss.
