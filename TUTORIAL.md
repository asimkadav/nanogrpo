# Fine-Tuning CLIP with Gradient-Reweighted Policy Optimization (GRPO)

## Introduction

Fine-tuning vision models like OpenAI's CLIP for image classification can be reframed as a reinforcement learning (RL) problem. Instead of just using standard supervised losses, we can use RL-style updates that incorporate behavioral feedback or non-differentiable objectives as a reward signal. Gradient-Reweighted Policy Optimization (GRPO) is a recent RL algorithm (a variant of PPO) that has been successfully applied to large language models and can be adapted to vision models.

GRPO introduces a clever way to avoid training a separate value function by using in-batch (or in-group) reward normalization. In this tutorial, we'll dive deep into how GRPO works for fine-tuning CLIP on classification tasks, using the asimkadav/nanogrpo implementation as a guide. We'll compare GRPO to standard PPO – covering entropy regularization, KL penalties, clipping, etc. – and walk through code snippets line by line.

## Background: PPO and RL Fine-Tuning Basics

Proximal Policy Optimization (PPO) is a popular policy-gradient RL algorithm often used in fine-tuning language models with human feedback (RLHF). PPO is known for stabilizing training via a clipped objective and typically uses an actor-critic approach. Key features of PPO include:

- **Policy Gradient with Advantage**: PPO maximizes an expected reward by policy gradient. It uses an advantage estimate A_i for each sample i, which indicates how much better the action was than some baseline. In vanilla PPO, the advantage is computed using a value function (critic) via methods like Generalized Advantage Estimation (GAE). The value function predicts future reward and serves as a baseline to reduce variance.

- **Clipped Surrogate Objective**: To avoid too-large policy updates, PPO doesn't directly maximize `r_i(θ) A_i` (where `r_i = π_θ(a_i|s_i)/π_θ_old(a_i|s_i)` is the probability ratio between new and old policy). Instead, it uses a clipped objective. For each action, it optimizes the minimum of (r_i A_i) and (clip(r_i, 1-ε, 1+ε) · A_i). This means if the policy tries to change probability too much (beyond a small ε, e.g., 0.2), the advantage is capped, removing incentive for further change. This clipping stabilizes training by preventing extreme updates.

- **Entropy Regularization**: PPO often adds an entropy bonus to the objective. Entropy H(p) = -∑_j p_j log p_j measures the policy's randomness. Maximizing entropy (or equivalently adding -α H as a negative bonus to the loss) encourages exploration by penalizing over-confident, low-entropy distributions. In practice, this helps prevent the model from prematurely collapsing to a very narrow set of outputs, which is important when learning from sparse feedback signals.

- **KL Divergence Penalty / Reference Model**: Especially in RLHF settings, it's useful to keep the fine-tuned policy from straying too far from the original model's behavior. PPO implementations for language models often include a KL-divergence penalty that measures how much the new policy's distribution π_θ deviates from a reference policy (usually the pre-trained model). By adding a term `-β D_KL(π_θ || π_ref)` to the reward or loss, the model is discouraged from "cheating" or extreme deviations, ensuring it remains aligned with its initial state. Intuitively, this prevents the model from exploiting the reward in unnatural ways, akin to an exam scenario where a student might try to game the system – the reference model acts like a regularizer to keep it honest.

- **Value Function (Critic)**: PPO's use of a learned value function as baseline means that during training, we update two networks: the policy (actor) and the value (critic). The critic tries to predict the total reward from a state so that `A = R - V(s)`. While effective, this doubles memory and compute usage, and if the reward is only given at the end of an episode (as is common in language tasks or one-step classification), training a token-level value function can be tricky.

### Why Think of CLIP Fine-Tuning as RL?

In a supervised image classification, we normally minimize cross-entropy with ground-truth labels. But by treating it as an RL problem, we have flexibility in defining the reward. For example, reward could be 1.0 for a correct classification and 0.0 for incorrect (mimicking accuracy), or something more nuanced.

RL also allows on-policy exploration: the model can sample outputs (class predictions) with probability proportional to its confidence, and learn from those outcomes, rather than always taking the single most likely class. This can help it escape local optima and improve classes it's unsure about.

However, applying vanilla PPO to CLIP has challenges: CLIP is a large model (ViT + text encoder) – training a separate value network of similar size for vision could be prohibitive. Also, in classification, the "episode" is just one step (predict a label for an image, then receive reward), so a value function might be overkill. This is where GRPO comes in.

## From PPO to GRPO: Removing the Value Function with In-Batch Normalization

Gradient-Reweighted (Group Relative) Policy Optimization (GRPO) was proposed to simplify PPO by dropping the value function critic in favor of a simpler baseline derived from a group of samples. The core idea: rather than using a learned value V(s) to estimate baseline reward, GRPO samples multiple outputs for the same input (a group) and uses their average reward as a baseline.

In effect, the advantage of each output is measured relative to others in the group, hence Group Relative. Concretely, for each input (say an image or a prompt) q, we generate a set of outputs {o_i}_{i=1}^G from the current policy (or a stale copy of it). We then score each output with a reward function, getting rewards r_1, r_2, ..., r_G. The baseline is simply the mean reward of those outputs, r̄ = (1/G)∑_i r_i. The advantage for each output i is A_i = r_i - r̄.

Often, we further normalize by the standard deviation as well, yielding:

A_i = (r_i - mean(r_{1...G}))/(std(r_{1...G}) + ε),

which makes the advantage a Z-score within the group. GRPO then plugs these advantages into the usual PPO surrogate loss (with clipping), but without needing any value network.

### Why This Matters for Vision Models Like CLIP

Vision tasks often have simpler reward structures (e.g., classification accuracy) and shorter episodes (one decision per image). Training a huge critic network to predict reward for each image is not only memory-intensive, it may be unnecessary. CLIP already provides a strong representation of images; using in-batch relative rewards leverages that the model can "compete against itself" on each input.

This reduces overhead and the risk of value function misestimation (which can destabilize PPO if the value network lags behind the policy). GRPO sidesteps that by assigning the final normalized reward to every token/action in the sequence – for image classification, this just means the single prediction gets that advantage.

### Group vs Batch Normalization

In GRPO for language tasks, a "group" means multiple responses to the same prompt. For image classification, we can take two approaches:

- **Per-Image Groups**: For each image, sample multiple label predictions from the model's output distribution. For example, have the model produce k candidate labels (by sampling from the softmax over classes k times). Then compute reward for each (1 if correct, 0 if wrong, in a simple case). Now treat those k samples as a group: compute the mean reward and advantages relative to that.

- **Batch-Level Normalization**: Alternatively, if sampling multiple outputs per image is computationally expensive, one can use the entire batch of images as the group for normalization. In this heuristic, for a batch of N images (each with one sampled prediction), we normalize rewards across the batch: A_j = (r_j - r̄_batch)/std(r_batch).

## Code Walkthrough: Computing Rewards and Loss in clip_grpo.py

Below, we step through a simplified version of what the core training loop in clip_grpo.py might look like:

```python
# Assume we have: 
# model - the CLIP model (with image & text encoders and a logit_scale parameter)
# ref_model - a fixed copy of the original CLIP (for KL penalty)
# images, labels - a batch of training images and their true labels
# config.hard_rewards (bool), config.entropy_coeff, config.kl_coeff, config.clip_eps, etc. as hyperparams

# 1. Encode image and text inputs
image_feats = model.encode_image(images)             # shape: [batch_size, image_feat_dim]
text_feats = model.encode_text(model.class_texts)    # precomputed text embeddings for each class label
logits = model.logit_scale.exp() * (image_feats @ text_feats.T)  # shape: [batch_size, num_classes]

# 2. Compute log-probabilities for the policy
log_probs = logits.log_softmax(dim=-1)  # log_probs[i, c] = log probability of class c for image i

# 3. Sample an action (class) for each image, possibly multiple samples per image
if config.num_samples_per_image > 1:
    # Sample k actions for each image i
    actions = [torch.multinomial(logits[i].softmax(dim=-1), num_samples=config.num_samples_per_image, replacement=True) 
               for i in range(len(images))]
    # actions is a list of length batch_size, each an array of k sampled class indices
else:
    # Sample one class per image from the probability distribution
    actions = torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(1)  
    # actions shape: [batch_size], each entry is the sampled class index

# 4. Compute rewards for each sampled action
if config.hard_rewards:
    # Hard reward: 1 if predicted class matches ground truth, 0 otherwise
    if config.num_samples_per_image > 1:
        rewards = [(sampled_classes == labels[i]).float() for i, sampled_classes in enumerate(actions)]
        # 'rewards' is a list of tensors, each of shape [k] for each image's k samples
    else:
        rewards = (actions == labels).float()  # tensor of shape [batch_size]
else:
    # Soft reward: use the model's probability of the correct class as a reward signal
    if config.num_samples_per_image > 1:
        rewards = []
        for i, sampled_classes in enumerate(actions):
            # probability of correct class for image i in each sample
            correct_prob = logits[i].softmax(dim=-1)[labels[i]]  
            rewards.append(torch.tensor([correct_prob.item() for _ in sampled_classes]))
    else:
        # reward is the probability the model assigned to the true label
        rewards = (log_probs.gather(1, labels.unsqueeze(1)).exp()).squeeze(1)  # shape [batch_size]

# 5. Convert rewards list to tensor and compute advantages via in-batch or in-group normalization
if config.num_samples_per_image > 1:
    # Stack rewards for each image group
    # rewards_list is list of length batch_size, each a tensor of shape [k]
    rewards_list = rewards  
    advantages_list = []
    for r in rewards_list:
        if r.std(unbiased=False) < 1e-6:
            # If all rewards in the group are identical (e.g., all 0s or all 1s), 
            # then no advantage (or we could skip update). 
            adv = torch.zeros_like(r)
        else:
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
        advantages_list.append(adv)
    # advantages_list is a list of tensors [k] per image.
    advantages = torch.cat(advantages_list)  # flatten all samples in batch together for loss computation
    logp_actions = [] 
    for i, sampled_classes in enumerate(actions):
        # gather log-probs of each sampled class from log_probs
        logp = log_probs[i, sampled_classes]  # tensor of shape [k] for image i's samples
        logp_actions.append(logp)
    logp_actions = torch.cat(logp_actions)  # concatenated log-probs for each sampled action in the batch
else:
    # Single sample per image: compute batch-wise advantage
    rewards = rewards  # tensor shape [batch_size]
    if rewards.std(unbiased=False) < 1e-6:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    logp_actions = log_probs[range(len(images)), actions]  # log-prob of the action actually taken for each image

# 6. Compute policy loss with PPO clipping
logp_old = logp_actions.detach()  # treat current policy (before update) as old policy
# Probability ratio for each taken action
ratio = (logp_actions - logp_old).exp()   # = exp(new_logp - old_logp)
pg_loss_1 = -advantages * ratio
pg_loss_2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
policy_loss = torch.mean(torch.max(pg_loss_1, pg_loss_2))

# 7. Entropy regularization (to encourage exploration)
probs = log_probs.exp()  # convert log-probs to probabilities
entropy = -(probs * log_probs).sum(dim=1).mean()   # average entropy over batch
entropy_loss = - config.entropy_coeff * entropy    # (will be added to total loss)

# 8. KL divergence against reference model (to keep policy from drifting too far)
with torch.no_grad():
    ref_log_probs = (ref_model.logit_scale.exp() * (image_feats @ ref_model.text_feats.T)).log_softmax(dim=-1)
kl_div = (probs * (log_probs - ref_log_probs)).sum(dim=1).mean()  # mean KL divergence for this batch
kl_loss = config.kl_coeff * kl_div

# 9. Total loss and optimization step
total_loss = policy_loss + kl_loss + entropy_loss
total_loss.backward()
optimizer.step()
```

### Explanation of Key Components

1. **Encoding and Logits**: We pass images through CLIP's image encoder and compare to text embeddings of class names. The logit_scale parameter controls the "peakedness" of the distribution.

2. **Sampling Actions**: We sample class predictions according to the model's current distribution to enable exploration.

3. **Hard vs Soft Rewards**: 
   - Hard rewards: Binary success indicator (1.0 for correct, 0.0 for incorrect)
   - Soft rewards: Using the model's probability for the correct class as a continuous reward signal

4. **Reward Normalization (Advantages)**: The core GRPO trick - using in-batch or in-group statistics to normalize rewards:
   - For multiple samples per image: Advantages are normalized within each image's group
   - For single samples: Batch-wise normalization is used
   - Zero-variance handling is critical for stability

5. **Policy Gradient Loss with Clipping**: Implements the PPO clipped surrogate objective to prevent too-large updates.

6. **Entropy Regularization**: Calculates policy entropy and adds a bonus to encourage exploration.

7. **KL Divergence to Reference**: Penalizes deviation from the original pre-trained CLIP to maintain its general knowledge.

8. **Combined Loss and Optimization**: Balances policy improvement with regularization.

## LoRA vs Full Fine-Tuning in GRPO

Fine-tuning CLIP with GRPO can be done in two ways:

### Full Fine-Tuning

- Updates all parameters of CLIP (image encoder, text encoder, logit scale)
- Maximum flexibility to adapt representations
- Computationally heavy and risks overfitting
- May require careful tuning of learning rates
- KL penalty becomes crucial to prevent unwanted drift

### LoRA Fine-Tuning

- Only updates small adapter weights (original weights stay fixed)
- Benefits:
  - Stability: Original knowledge is preserved
  - Efficiency: Fewer parameters to update means faster training, less memory
  - Natural KL regularization: Model can't stray too far from reference
- Limitation: May not capture complex shifts if the task requires it

In practice, LoRA is often a good first choice for RL fine-tuning of large models like CLIP. You could also mix approaches: fully fine-tune final layers or logit_scale, but use LoRA on earlier layers.

## Using Real Behavioral Data as Rewards

One of the key advantages of the RL approach is incorporating real-world feedback signals:

- **Click-through rates or heatmaps**: Which images users click on or how long they view them
- **Pairwise preferences**: When users prefer one model's output over another's
- **Quality annotations beyond correctness**: When multiple labels could apply, but some are more precise

### Handling Sparse/Noisy Feedback

- **Sparse rewards**: Not every image will have an explicit signal
  - Use a mix of supervised signals and feedback-based rewards
  - Warm-start with supervised learning before GRPO
  - Shape rewards to provide gradient even for partial success

- **Noisy feedback**: User behavior can be inconsistent
  - The KL penalty helps prevent overreacting to noise
  - Entropy regularization maintains reasonable alternatives
  - Consider smoothing rewards (e.g., running averages of click rates)

- **Evaluation**: Monitor both reward optimization and original performance metrics to ensure balance

## Implementation Nuances

Several important details ensure stable and effective GRPO training:

### Logit Scaling

- CLIP's logit_scale controls the "temperature" of predictions
- Higher values create more peaked distributions (lower entropy)
- During GRPO, consider:
  - Freezing logit_scale to a moderate value to maintain exploration
  - Monitoring for entropy collapse if logit_scale grows too large
  - Using separate scales for policy and reference models

### Gradient Handling

- Use gradient clipping for numerical stability
- Balance exploration-exploitation with entropy weight
- Prevent zero-division with small epsilon terms

### Reward Design

- Consider combining accuracy rewards with user preference signals
- Normalize rewards within groups to get proper relative advantages
- Handle zero-variance cases gracefully (e.g., when all samples are right/wrong)

## Conclusion

Gradient-Reweighted Policy Optimization offers a powerful yet simpler framework for fine-tuning vision models like CLIP using RL-style updates. By avoiding a learned value function and normalizing rewards in-batch, GRPO makes the fine-tuning process more efficient in terms of memory and implementation complexity.

The key advantages of GRPO for CLIP fine-tuning include:

1. **Simplicity**: No separate value network to train
2. **Memory efficiency**: Approximately half the memory of PPO
3. **Flexibility in rewards**: Can incorporate user feedback, clicks, or any measurable signal
4. **Stable training**: Normalization and clipping prevent extreme updates

GRPO is particularly well-suited for incorporating non-traditional feedback signals into model training. Whether it's human preference data, click feedback, or other forms of weak supervision, GRPO can optimize the model towards those goals while using the reference model and entropy to keep it from going astray.

For a vision practitioner, thinking in terms of RL for fine-tuning opens up new possibilities beyond standard supervised learning. The conceptual payoff is that you're not limited to predicting fixed labels – you can fine-tune your model to maximize any behavior you can measure.

In summary, GRPO fine-tuning of CLIP marries the best of both worlds: the strong prior of a pre-trained model and the nuanced objectives of RL. It avoids the heavyweight critic of PPO, normalizes away a lot of extraneous variance, and provides a stable platform to inject human and task-specific insights into vision models.

## Further Reading

- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
- OpenAI, "Learning from Human Preferences"
- Andrej Karpathy, "Deep Reinforcement Learning: Pong from Pixels"
- CLIP paper: "Learning Transferable Visual Models From Natural Language Supervision" 