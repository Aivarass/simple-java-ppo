# PPO from Scratch in Java

A complete implementation of Proximal Policy Optimization built entirely from scratch using vanilla Java. No machine learning libraries. No external dependencies beyond JUnit for running the tests.

## Overview

This project demonstrates a working PPO implementation applied to a simple game environment. The focus is on the algorithm itself, showing all components of modern policy gradient methods in readable, documented code.

## PPO Components

### Neural Network

A shallow actor critic network with shared trunk architecture. The trunk is a single hidden layer with tanh activation. Two separate heads branch from the trunk. The actor head produces action logits which pass through softmax to get probabilities. The critic head produces a scalar value estimate.

Weight initialization uses Xavier scaling appropriate for tanh activations.

### Trajectory Collection

Rather than updating after each environment step, the agent collects an entire trajectory of experiences first. Each transition stores the state, action taken, probability of that action at selection time, reward received, next state, and terminal flag.

### Advantage Estimation

After collection, advantages are computed for every transition using TD(0). The key detail is that all advantages are computed using the same critic parameters before any updates occur. This frozen snapshot provides consistent training signal across the batch.

Advantages measure how much better the outcome was compared to what the critic predicted.

### Importance Sampling Ratio

The ratio compares the current policy probability to the original probability when the action was taken.

```
ratio = pi_new(action | state) / pi_old(action | state)
```

This ratio enables reusing experiences multiple times. If the policy now thinks the action is more likely, the ratio exceeds one. If less likely, the ratio falls below one.

### PPO Clipping

The core PPO mechanism constrains how much the policy can change per update. When the ratio goes outside the range defined by epsilon, the policy gradient is zeroed out.

```
if advantage >= 0:
    clip when ratio > 1 + epsilon
else:
    clip when ratio < 1 - epsilon
```

This prevents catastrophic policy updates that could destabilize training.

### Multiple Epochs

Each trajectory is reused for multiple passes of updates. This improves sample efficiency. The clipping mechanism ensures that as the policy drifts from its collection time values across epochs, updates become progressively more conservative.

### Entropy Regularization

An entropy bonus encourages exploration by penalizing overly deterministic policies. The entropy gradient flows even when the policy gradient is clipped, maintaining some exploration throughout training.

## Implementation Details

The update method handles critic, actor, and trunk weight updates in sequence.

The critic updates using the advantage signal to improve value predictions.

The actor computes importance weighted gradients, applies PPO clipping, and combines with entropy gradient.

The trunk receives backpropagated gradients from both heads.

Weight snapshots are taken before updates to ensure gradients use pre update values.

## Configuration

Typical hyperparameters included in the implementation.

Actor and critic learning rates around 0.04 to 0.06.

PPO epsilon of 0.2 giving a clipping range of 0.8 to 1.2.

Four epochs per trajectory.

Discount factor of 0.99.

Entropy coefficient of 0.005.

Advantage clipping between negative five and positive five for numerical stability.

## Project Structure

TinyActorCriticANN contains the neural network, forward pass, and PPO update logic.

StateVectorizer normalizes raw state features to the range suitable for neural network input.

The engine class handles environment simulation, trajectory collection, and training loop orchestration.

Data classes in the model package hold transitions, episode statistics, and step results.

## Technical Notes

All matrix operations are explicit loops over arrays. There is no linear algebra library.

Softmax uses the max subtraction trick for numerical stability.

Backpropagation through the shared trunk correctly accumulates gradients from both heads.

The implementation prioritizes clarity over performance. Each concept maps directly to textbook descriptions of PPO.

## Potential Extensions

Generalized Advantage Estimation would provide better variance reduction than TD(0).

Mini batch sampling within epochs is common in larger scale implementations.

Learning rate annealing can improve convergence in later training.

Separate networks for actor and critic is an alternative to the shared trunk design.
