# A Practical Guide to Fine-Tuning a Language Model for German–English Podcast Translation

*Written for someone who understands Python and machine learning basics, but has not fine-tuned a language model before.*

---

## Why a 1.5B Model Can Outperform a 4B Model on a Narrow Domain

It seems counterintuitive: how can a smaller model do better than a larger one? The answer lies in what "performance" means in different contexts.

A large model like a 4B or 7B parameter network has spent enormous compute developing a broad, general understanding of language. It has seen academic papers, parliamentary debates, novels, forum posts, and news articles in dozens of languages. This breadth makes it excellent at tasks it has never been explicitly trained on. But that same breadth comes at a cost: the model has learned a distribution that covers everything, which means its prior belief about "what a good German–English translation looks like" is very diffuse. It knows that translations can be formal or informal, literal or liberal, concise or verbose, and it has learned to hedge.

When you fine-tune on a narrow domain — specifically, spoken podcast German translated into natural conversational English — you are not teaching the model new knowledge. You are updating its probability distribution to concentrate on the register, vocabulary, and phrasing patterns that characterise your specific task. The model's prior is effectively replaced by a much more focused posterior. A 4B model that has not been fine-tuned has more capacity, but that capacity is spread across task space far more thinly than your fine-tuned 1.5B model.

The theoretical basis for this is Bayesian: fine-tuning is a form of posterior inference, where the likelihood comes from your training data and the prior comes from the pre-trained weights. A smaller, well-targeted model fine-tuned on the right data can have a sharper posterior for your specific task than a larger model with a flat prior. This is not just a theoretical claim — it is consistently observed empirically in domain-specific translation, medical NLP, and legal document processing.

There are limits, of course. If the task requires world knowledge that the smaller model never saw during pre-training (unusual proper nouns, technical jargon not in the training corpus, very long-range dependencies), the larger model will maintain an advantage. The sweet spot for fine-tuning is tasks where register and style matter more than encyclopedic knowledge — which is precisely the case for podcast translation.

---

## What SFT Is Doing at the Level of Gradient Updates

Supervised Fine-Tuning sounds complicated, but at its core it is doing something very simple: for each German segment in your training data, you show the model the German input and the reference English translation, and you adjust the model weights to make the reference translation slightly more probable.

More concretely, the training process works like this. The model takes the prompt (the German segment plus the surrounding instruction text) and produces a probability distribution over its vocabulary for each position in the output sequence. The reference English translation specifies exactly which words should be chosen. The cross-entropy loss measures how surprised the model is by the reference: if the model already assigns high probability to the reference words, the loss is low; if it assigns low probability, the loss is high.

The gradient of this loss with respect to each model parameter tells us, for every weight in the network, whether increasing or decreasing it would make the reference translation more likely. Gradient descent takes a small step in the direction that increases this likelihood. After thousands of steps across thousands of examples, the model's weights have shifted to make the entire class of "conversational German → natural English translations" more probable.

LoRA (Low-Rank Adaptation) does not change this process conceptually — it just constrains *which* weights are updated. Rather than updating every one of the model's 1.5 billion parameters, LoRA introduces small "delta matrices" into the attention layers, and only updates those. This means 99% of the original model's knowledge is preserved exactly; only the delta matrices are trained. The result is an adapter that is perhaps 50 MB in size but meaningfully shifts the model's translation behaviour. This is both memory-efficient and fast to train.

**Why SFT is necessary before RL.** Reinforcement learning works by generating candidate outputs, scoring them, and updating the model to prefer high-scoring outputs. But this requires the model to already be generating *something reasonable* — if the base model's translation outputs are too far from acceptable English, the reward signal has nothing to work with. A model that generates random text gets uniformly mediocre rewards, and the gradient signal is uninformative. SFT focuses the model's output distribution so that GRPO has a meaningful starting point: candidates that are mostly correct translations that differ in fluency and register, rather than candidates that span the entire space of possible English text.

---

## What GRPO Is, How It Differs from PPO, and Why It Suits This Project

Group Relative Policy Optimisation was developed as a simpler, more memory-efficient alternative to Proximal Policy Optimisation (PPO). To understand the difference, it helps to understand what PPO requires.

PPO is the classic approach to reinforcement learning from human feedback (RLHF). It involves training *four* separate models simultaneously: the policy model (the one being trained), a frozen reference policy (used to prevent the policy from drifting too far), a reward model (trained to predict human preferences), and a value model (trained to estimate the expected reward from a given state). The value model is necessary because PPO computes "advantage" — how much better a given action was compared to the expected baseline. Training four models at once is expensive in both compute and memory.

GRPO eliminates the value model and the reward model. Instead of a separately trained value function, it uses the group statistics of the reward: for each input, it generates G candidate outputs (translations, in our case), scores them all using a reward function, and computes advantages relative to the group mean. If one translation scores 0.8 and the group mean is 0.6, that translation's advantage is positive and the model is pushed to produce more like it. If a translation scores 0.4, the model is pushed away from it.

This is enormously more practical for a small project. You do not need to train and maintain a separate reward model — you just need a callable reward function. In our case, that function is a call to an LLM judge via the OpenRouter API. And because you do not need a value model, memory requirements drop substantially: instead of four models in memory, you need roughly one and a half (the policy, plus the frozen reference for KL penalty computation).

**Why GRPO is better suited to this project than PPO.** The honest answer is: because you have a Colab session with limited memory and no engineering team. PPO at this scale would require careful orchestration of four models, possibly across multiple GPUs, with complex synchronisation. GRPO requires one model, one reward function, and a for-loop. The quality difference between PPO and GRPO for translation is not established to be large enough to justify the additional complexity at this scale.

---

## Why Translation Is a Good Task for LLM-as-Judge Reward

The ideal reward signal for reinforcement learning is *verifiable* — something you can check with certainty. For mathematical reasoning, you can verify that the answer is correct. For code, you can run tests. For translation, there is no ground truth in that sense: a German sentence can be translated in dozens of equally valid ways, and whether any given translation is "correct" depends on register, context, and the target audience's expectations.

This is precisely why LLM-as-judge works well for translation. A large language model can evaluate a translation against the original on dimensions that have no simple algorithmic equivalent: does this sound like something a German speaker would actually say? Does the English feel natural in a conversational context? Is the meaning preserved despite paraphrase? These are judgements that require broad linguistic knowledge and a model of natural language use — exactly what a large frontier model has.

The alternative — using COMET as the RL reward — has a fundamental problem: COMET is a reference-based metric, meaning it compares the model's output against a fixed reference translation. If the model learns to produce outputs that are similar to the reference according to COMET, it is simply learning to mimic the reference, which is a more expensive version of SFT. COMET-QE (the reference-free variant) avoids this but is less interpretable and harder to tune.

**The risks of LLM-as-judge reward.** This approach has genuine failure modes that you must watch for.

The most serious is *positional and stylistic bias*. LLM judges can systematically prefer certain surface features: longer outputs, more formal register, hedged phrasings, or outputs that begin with certain words. If your judge model has been trained with RLHF to prefer certain writing styles, your translation model will be rewarded for mimicking those styles, not for genuine quality.

A related risk is *inconsistency*. Two calls to the same LLM judge with the same input may produce different scores, especially if the temperature is not set to 0. This adds noise to the reward signal. We mitigate this by using `temperature=0.0` for all reward calls.

Finally, there is the risk of *gaming*. A sufficiently capable policy model may discover that certain output patterns reliably produce high scores from the judge — not because those outputs are good translations, but because they happen to match the judge's implicit preferences. Watching the score trajectory over training (it should rise gradually, not spike) and monitoring output length (it should remain stable) are the primary defences against this.

---

## What Overfitting Looks Like in SFT for Translation

Overfitting in SFT is subtler than in standard classification tasks. You cannot just look at training accuracy and validation accuracy and see them diverge, because the task is generative and there is no single correct output.

The clearest signal is the **validation loss plateau**. During healthy training, both training and validation loss fall in tandem. When overfitting begins, training loss continues to fall but validation loss stops decreasing and may start to rise. This is the model memorising specific training examples rather than learning general translation patterns.

A second signal is **chrF++ behaviour** on the validation set. In a healthy run, chrF++ rises throughout training and then plateaus. In an overfitting run, chrF++ may begin to fall *before* the validation loss rises, because the model's output distribution narrows — it starts producing translations that are very similar to each other, reflecting the specific phrasings in the training data rather than exploring the space of valid translations.

Qualitatively, overfitted translation models produce output that sounds like it was copied from a textbook. The sentences are grammatically impeccable but oddly similar to each other in structure, as if the model has learned a template rather than a strategy. You might notice that the model always places time expressions at the end of sentences, or always uses "furthermore" when the original uses "und außerdem" — patterns that appear frequently in the training data but are too rigid for a general-purpose translation system.

**When to stop SFT.** Use the validation COMET score as the primary criterion. We save checkpoints every 500 steps and load the best checkpoint at the end of training. If the best checkpoint is not the final one, that tells you something useful: training ran for too long, and you should reduce the number of epochs in future runs. More practically, if chrF++ has not improved by more than 0.5 points in 2,000 steps, stop. The model is done learning from this data.

---

## What Reward Hacking Looks Like in RL for Translation

Reward hacking occurs when the model finds a strategy that maximises the reward signal without actually improving translation quality. It is the machine learning equivalent of a student who memorises the grading rubric without understanding the material.

For translation specifically, the classic failure modes are:

**Verbosity exploitation.** If the LLM judge tends to reward longer, more detailed translations, the model will learn to pad its outputs — adding phrases like "it is worth noting that" or "in other words" that were not in the original German. The length monitoring plot in notebook 03 is your primary guard against this: if mean output length starts growing linearly after step 100, suspect verbosity hacking.

**Formality drift.** Some LLM judges have been trained to prefer formal, "professional-sounding" English. A model being trained with such a judge will learn to produce translations that sound like a formal report rather than a podcast. You would notice this qualitatively before any metric flags it — the translations will start sounding stiff and over-hedged.

**Repetitive safe phrases.** The model may learn that certain high-frequency phrasings always receive acceptable scores — something like always translating "das ist interessant" as "that is interesting" regardless of context, because it reliably gets a score of 0.6. Monitoring the diversity of outputs (the `set(candidates)` check in notebook 03) catches this.

**Score saturation without quality improvement.** The most insidious form: the judge starts awarding 0.85–0.9 to most outputs, but the actual translation quality has not changed. This can happen if the policy model's distribution converges to a mode that happens to match the judge's preferences. The fix is to periodically run COMET (the fixed metric) on the validation set alongside the RL reward — if COMET is flat while the RL reward is rising, reward hacking is likely underway.

---

## How to Interpret COMET Scores in Practice

COMET (specifically `Unbabel/wmt22-comet-da`) outputs a score in roughly the range [0.7, 0.95] for translation systems that are actually producing reasonable output. Raw numbers below 0.6 indicate a poorly functioning system; numbers above 0.9 are achieved only by top WMT competition systems.

**A difference of 1 COMET point (0.01 on the 0–1 scale)** is statistically significant on a test set of 500 segments with variance typical of spoken-language MT. It represents a difference that a trained evaluator would detect if reading the outputs side by side, but that most casual readers would not notice in a single sentence. Think of it as the difference between "this translation is fine" and "this translation is subtly better — the phrasing is a bit more natural."

**A difference of 5 COMET points (0.05)** is clearly perceivable. At this level, one system's outputs read noticeably more fluently or more accurately than the other's. If your fine-tuned model achieves a 5-point COMET improvement over the baseline, that is a genuine, human-detectable quality gain. If you handed a native German speaker the German source and both translations, they would reliably identify the better one.

**A difference of more than 10 points (0.10)** represents a dramatic quality gap — the kind you see between an untuned base model and a professionally tuned translation system, or between MT and human translation.

In our context, a realistic target is a **3–6 COMET point improvement** over the untuned Qwen2.5-1.5B base model. If you achieve more than 6 points, your training was particularly successful. If you achieve less than 1 point, something went wrong — check the data quality, the prompt format consistency, and the reward hacking indicators before concluding that the approach is flawed.

---

## The Trade-offs of Choosing a 1.5B Model

Choosing to work with a 1.5B model is a pragmatic decision driven by hardware constraints, but it comes with genuine trade-offs that are worth being honest about.

**What the model will likely do well.** Short to medium-length conversational sentences (15–80 tokens of German) are well within the model's ability after fine-tuning. German idioms that appear in the training data will be translated correctly. The model will maintain consistent register (conversational rather than formal) if the training data is well-curated. Common discourse markers ("also", "nämlich", "übrigens") will be handled appropriately.

**What the model will likely struggle with.** Very long compound sentences — a feature of spoken German that is relatively rare in English — will sometimes be translated with awkward structure because the model needs to restructure the syntax significantly. Named entities that appear rarely in the training data (new companies, niche academic terms, cultural references specific to Germany) may be passed through or mistranslated. Highly context-dependent meaning (irony, understatement, and some register shifts) requires more world knowledge than a 1.5B model reliably has.

**What could be done next if quality is not sufficient.** The most impactful interventions, roughly in order of effort:

1. **Add domain-specific data.** Even 1,000 German podcast segments with manually verified English translations would dramatically improve performance. This is the single highest-leverage action available.

2. **Upgrade to a 3B or 7B model.** Qwen2.5 is available at 3B and 7B with the same licence and MLX compatibility. A 3B model in 4-bit quantisation uses approximately 2 GB of unified memory, which fits on an 8 GB M1 Mac. The quality improvement from 1.5B to 3B on translation is typically 3–8 COMET points on the same training data.

3. **More GRPO steps.** RL improvements tend to be incremental but cumulative. If you stopped at 500 GRPO steps, trying 1,500–2,000 often yields further gains, provided the reward signal is not saturating.

4. **Better reward prompt.** The LLM judge prompt is a lever you can adjust without retraining. Adding a third evaluation dimension (register appropriateness, or specific German idioms that should not be translated literally) can redirect the reward signal towards quality aspects that simple adequacy + fluency misses.

---

## How to Decide Whether More RL Steps Are Helping or Hurting

The guiding principle is to track multiple metrics simultaneously, not just the RL reward. The RL reward is endogenous — the model is actively being trained to increase it, so it will always trend upward. The question is whether that upward trend reflects genuine quality improvement or reward hacking.

Run your fixed evaluation metrics (COMET, chrF++) on the validation set every 100 GRPO steps. If they track the RL reward — rising as the reward rises — you are making genuine progress. If they plateau while the RL reward keeps climbing, the model is exploiting the judge. If the fixed metrics start falling, stop immediately and restore the last checkpoint where they were at their peak.

A second useful heuristic is to inspect the actual outputs at different checkpoints. Read five outputs from step 100, five from step 300, and five from step 500. If the quality appears to be improving subjectively, continue. If the outputs at step 500 feel similar to step 300 but slightly more verbose or slightly more formulaic, that is a signal to stop.

The general shape of a healthy GRPO training run: the reward rises quickly for the first 100–200 steps (the model is learning the basic rules of the reward prompt), then rises more slowly as it refines the finer points, then plateaus as the KL constraint limits further improvement. If the reward is still rising steeply at step 500, you might have room for more steps. If it plateaued at step 200, more steps are wasted compute.

---

## What Happens During LoRA Merging and Quantisation

**LoRA merging** is a simple matrix operation. The LoRA adapter stores two small matrices A and B for each adapted layer. During forward pass, the adapter contribution is computed as (BA) × input, where the product BA is a low-rank approximation of the full weight update. When you call `merge_and_unload()`, the library computes the full weight update (α/r × BA) and adds it to the frozen base weights. The result is mathematically equivalent to the adapted model but stored as a single set of full-rank matrices, with no PEFT library required at inference time.

Nothing is lost in this operation — it is exact arithmetic, not an approximation. The merged model produces identical outputs to the unmerged model + adapter combination (modulo floating-point rounding on some hardware, which is negligible).

**Quantisation** is a different story. When mlx-lm converts the merged bfloat16 model to 4-bit integers, each weight is no longer stored with 16 bits of precision but with 4 bits — one of only 16 possible values. The quantiser maps each weight to the nearest of those 16 values, introducing rounding error. The error is not large in absolute terms (the quantiser is designed to minimise it using calibration statistics), but it is real.

**Should you expect the MLX model to behave identically to the Colab model?** No. Here is what to expect:

The outputs will be very similar for common, high-frequency sentence patterns. The most likely word at each step usually remains the same after quantisation, so for clear, unambiguous translation inputs, the output will often be identical.

The outputs will differ noticeably for longer generations (more than 60–70 tokens) where small probability differences compound. In these cases, the quantised model may choose a different synonym, restructure a clause slightly differently, or split a sentence differently.

The quality gap is approximately 0.5–1 COMET points — measurable, but below the threshold of human perception in side-by-side comparison for most listeners. If you specifically need the MLX model to match the Colab model's outputs exactly (for example, if you are using the outputs as input to another system), use 8-bit quantisation, which reduces the quality gap to less than 0.2 COMET points.

The practical implication: trust the Colab evaluation results (notebook 04) as your quality baseline, and treat the MLX model as approximately equivalent. If you notice specific translation errors on your Mac that were not present in the Colab version, this is likely a quantisation artefact, and switching to 8-bit will usually resolve it.

---

## Glossary

**Adequacy.** In machine translation evaluation, the degree to which a translation correctly conveys the meaning of the original. A translation can be fluent (sounds natural) but inadequate (wrong meaning), or adequate (correct meaning) but disfluent (awkward).

**Adapter.** A small set of trainable parameters added to a frozen pre-trained model. See also: LoRA.

**Advantage (in RL).** How much better an action was compared to the expected baseline. In GRPO, the advantage of a translation is how much its reward exceeds the group mean.

**Apple Silicon.** Apple's line of ARM-based chips (M1, M2, M3, M4) used in recent Macs. They use a unified memory architecture where the CPU and GPU share a single pool of memory, which has implications for LLM inference.

**Base model.** The pre-trained model before any fine-tuning. In this project, `Qwen/Qwen2.5-1.5B-Instruct`.

**bfloat16.** A 16-bit floating-point number format used in modern neural network training. Has the same exponent range as float32 but less mantissa precision. Standard for TPU training.

**BLEU.** Bilingual Evaluation Understudy. A classic MT metric based on precision of n-gram overlap between the hypothesis and reference translation. Widely reported for historical comparability, but has low correlation with human judgement compared to neural metrics.

**Byte-pair encoding (BPE).** A tokenisation algorithm that splits words into subword units. Qwen2.5 uses a BPE tokeniser, which means that German compound words like "Bundesgesundheitsministerium" are split into multiple tokens.

**causal LM.** A language model trained to predict the next token given all previous tokens. Qwen2.5 is a causal LM. Contrasted with masked LMs (like BERT) which predict masked tokens given context on both sides.

**chrF++.** Character n-gram F-score with word bigrams. An MT metric that operates at the character level, making it sensitive to morphological variation (important for German). The `++` variant adds word-level bigrams. Correlates better with human judgement than BLEU.

**COMET.** Crosslingual Optimised Metric for Evaluation of Translation. A family of neural MT metrics trained on human quality assessments. `Unbabel/wmt22-comet-da` is the reference model used in this project.

**CoVoST-2.** A multilingual speech-to-text translation corpus derived from Mozilla Common Voice, covering translations from 21 languages into English. CC0 licensed.

**Cross-entropy loss.** The standard loss function for language model training. Measures how surprised the model is by the reference token at each position — equivalent to the negative log-probability of the reference given the context.

**Fine-tuning.** Updating a pre-trained model's weights on task-specific data. Contrasted with training from scratch or zero-shot inference.

**Fluency.** In MT evaluation, the degree to which a translation sounds natural in the target language, independent of whether it is accurate. A translation can be fluent but wrong.

**Gradient descent.** The optimisation algorithm used to train neural networks. At each step, the gradient of the loss with respect to all model parameters is computed, and each parameter is adjusted slightly in the direction that reduces the loss.

**GRPO.** Group Relative Policy Optimisation. A reinforcement learning algorithm that estimates advantages by comparing rewards within a group of candidates generated from the same input. Eliminates the value model required by PPO.

**Held-out test set.** A subset of the data that is never used during training or hyperparameter tuning, reserved exclusively for final evaluation. Also called the "sacred" test set in this project.

**KL divergence (Kullback-Leibler divergence).** A measure of how different one probability distribution is from another. In RL fine-tuning, the KL penalty constrains the trained model to stay close to the reference model, preventing reward hacking and catastrophic forgetting.

**LoRA (Low-Rank Adaptation).** A parameter-efficient fine-tuning technique that introduces small trainable matrices (adapters) into existing layers, keeping the base model weights frozen.

**LLM judge.** Using a large language model to evaluate the quality of another model's outputs. Common in RL fine-tuning and evaluation when the task is too complex for a simple scoring function.

**MLX.** A numerical computation framework developed by Apple for Apple Silicon. Designed to exploit unified memory and supports lazy evaluation, operator fusion, and Metal GPU acceleration.

**mlx-lm.** A Python library built on MLX for running language model inference on Apple Silicon. Supports loading Hugging Face models and converting them to MLX format.

**Mode collapse.** A failure mode in generative models where the model produces the same output (or very similar outputs) for all inputs, losing diversity.

**MuST-C.** Multilingual Speech Translation Corpus. Contains TED Talk recordings aligned with transcriptions and translations. The closest available proxy for podcast speech.

**OpenSubtitles.** A large corpus of movie and TV subtitles aligned across languages. Informal register, useful for supplementing more carefully curated translation data.

**PEFT (Parameter-Efficient Fine-Tuning).** A family of techniques (including LoRA) for fine-tuning large models with far fewer trainable parameters than full fine-tuning.

**Policy model.** In RL, the model being trained. Contrasted with the reference policy (frozen) and the value model (also frozen).

**PPO (Proximal Policy Optimisation).** A reinforcement learning algorithm used in RLHF. Requires training a value model and a reward model alongside the policy. More expensive than GRPO but theoretically well-understood.

**Quantisation.** Representing model weights with fewer bits than the training precision. 4-bit quantisation reduces memory usage by approximately 8× compared to float32, at the cost of a small quality reduction.

**Register.** The style and formality of language appropriate to a context. Podcast speech is informal/conversational register. Parliamentary speech (like Europarl) is formal register. Register mismatch in training data is a major source of translation quality problems.

**Reward hacking.** When a model learns to maximise the reward signal without actually improving on the underlying task. The model "games" the reward function rather than improving genuinely.

**RLHF (Reinforcement Learning from Human Feedback).** Training a model using reinforcement learning with rewards derived from human preference judgements. GRPO is a variant that uses an LLM judge rather than direct human feedback.

**Safetensors.** A file format for storing model weights, developed by Hugging Face. Safer than PyTorch's pickle-based format (no arbitrary code execution) and supports memory mapping for faster loading.

**SFT (Supervised Fine-Tuning).** Training a model by maximising the likelihood of reference outputs given inputs, using standard gradient descent. The first stage of adaptation in this project.

**Temperature.** A parameter that controls the randomness of language model generation. Temperature 0 (or very close to 0) produces greedy, deterministic output. Temperature 1.0 samples proportionally to the model's raw probabilities. Temperature > 1.0 increases randomness; < 1.0 decreases it.

**TRL.** Transformer Reinforcement Learning. A Hugging Face library providing implementations of SFT, PPO, and GRPO training loops for language models.

**Unified memory.** The hardware architecture of Apple Silicon chips, where CPU and GPU share a single pool of memory. This means a language model can use all available system memory for weights and KV cache, without the overhead of CPU–GPU transfers.

**Unsloth.** A Python library that provides memory-efficient implementations of LoRA fine-tuning and GRPO for LLMs, with support for 4-bit quantisation on GPU.

**WMT (Workshop on Machine Translation).** An annual shared task at a major NLP conference. WMT provides standardised test sets and human evaluation data for MT, and the WMT metrics shared task benchmarks automatic metrics against human judgement.

---

*End of tutorial. The notebooks are designed to be read alongside this document — each major decision in the code has a "Decision note" that cross-references the reasoning described here.*
