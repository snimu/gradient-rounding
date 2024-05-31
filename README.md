# gradient-rounding

I round the gradient of an LLM during training, performing an ablation over the number of significant digits to round to.

My original motivation was thinking about scaling laws with the number of parameters.
I will first explain this motivation and its connection to gradient rounding, and then give the results of my experiments.

Here is the table of contents:

- [Original motivation](#original-motivation)
    - [Larger models live in higher-dimensional space](#larger-models-live-in-higher-dimensional-space)
    - [Different behaviors are connected more continuously in high- than in low-dimensional space, making SGD easier](#different-behaviors-are-connected-more-continuously-in-high--than-in-low-dimensional-space-making-sgd-easier)
    - [How would I test this?](#how-would-i-test-this)
- [Results](#results)
    - [Loss curves for different gradient rounding digits](#loss-curves-for-different-gradient-rounding-digits)
    - [L2 distance to losses of $M_{16}$](#l2-distance-to-losses-of-m_16)
    - [L2 distance to losses of $M_{max}$](#l2-distance-to-losses-of-m_max)
- [Summary](#summary)
- [Acknowledgements](#acknowledgements)

Note: I did this on a whim within a few days, and haven't really done any research prior.
If you have any information on the subject that you can share with me, please feel free to do so @omouamoua on X/Twitter.

## Original motivation

I was curios why larger models learn faster than smaller models.

You might think that this is not surprising; after all, a smaller model has less model capacity, so obviously it cannot learn as much as a larger one. However, this only explains why the smaller model stops improving after fewer datapoints than the larger model does. It does not explain why even early in training, when there is plenty of model capacity left for learning, the larger model is able to extract more information from the training data than the smaller model.

Put another way:

- You have a large and a small model
- You have a training dataset that is small enough that both models can easily learn the underlying function that generated the data
- But the large model will learn the function after fewer samples than the small model

The question is: why is that?

Here is my idle speculation:

> [!IMPORTANT]
> Larger models live in higher-dimensional space, where different behaviors are connected much more continuously than in lower-dimensional space. This makes it easier for stochastic gradient descent (SGD) to find the best weight configuration, because it is a form of local search.

Let's break this down.

### Larger models live in higher-dimensional space

Any deep neural network can be replaced by a shallow one with only one hidden layer.
The more parameters the original model has, the wider the shallow network, and therefore,
the higher-dimensional its internal representation.

### Different behaviors are connected more continuously in high- than in low-dimensional space, making SGD easier

The core of this argument is that SGD is local search. At every step, we want to keep the weights as similar to the previous weights as possible, to avoid unlearning the lessons from the previous step.

To make thinking about this easier, imagine that all weights are binary.

Then a model with $2$ weights can represent $4$ possible algorithms.
If you choose one of those, then compute a loss, you have to make a big jump in the weigths in order to change behavior,
towards weights with no obvious connection to the current one.
This makes local search pretty difficult.

So what if we have $1$ *billion* parameters?
This can represent $2^{1e9}$ algorithms (I think), many of which will only differ slightly.
Whatever algorithms is represented right now,
we can likely find an algorithm that is better, and is continuously connected to the current one in weight space,
so that we can take several small steps towards the better algorithm instead of one large jump.
This makes local search easier for SGD.

Why does increasing the number of weights even further continue to help?
It allows for a more fine-grained search.
If you want to change a small aspect of the current algorithm, but nothing else,
then if you have too few parameters, it might be difficult to find a corresponding set of weigths close to the current ones,
and you are forced to add some noise to the unchanged behavior if you want to take only a small step.
But if you have many parameters, all kinds of algorithms that are exactly like the current one
except in one or two aspects will be very close-by, and therefore easy to find by SGD.
In other words, with many parameters, you can learn from the current batch without unlearning the previous batches' contents, despite only making small changes.

### How would I test this?

The essence of the proposed explanation is that a small number of parameters restricts the search space for algorithms.
So to test the explanation, I need another way to restrict the search space to compare to.

My solution is to round the gradients to different numbers of significant digits.
I can now compare the "scaling laws" of gradient-rounding to those of parameter count.

## Results

Let's call the largest model I train $M_{max}$, for any number of significant digits,
and any model trained with gradients rounded to $16$ digits $M_{16}$, regardless of size.

I will give the results in three steps:

1. Plot the validation loss over the training tokens at a fixed model size, for different gradient rounding digits.
2. Plot the L2-distance between the loss-curve of models with gradients rounded to some number of digits and
    $M_{16}$ at the same size.
3. Plot the L2-distance between the loss-curves of all models to that of $M_{max}$, for the same gradient rounding digits.

### Loss curves for different gradient rounding digits

To get a feeling for the results, let's first plot the validation loss over the training tokens for the default model size:

![Validation loss over training tokens; default model size; gradient rounded to 1-16 significant digits](results/images/val_loss_vs_token_from_percentage_0.0_depth_8_width_384_num_heads_1.png)

We can immediately see that up to $5$ digits, the gradient rounding very strongly degrades performance. Beyond that, it is more difficult to see in this plot, because all the loss curves starting from $6$ digits are one big blob.

So let's plot only those curves, zoomed in to the later tokens:

![Validation loss over training tokens; default model size; gradient rounded to 6-16 significant digits](results/images/val_loss_vs_token_from_percentage_0.7_depth_8_width_384_num_heads_1_digits_6_to_16.png)

Clearly, rounding to $6$ digits still degrades performance; but beyond that, it is much less clear.
It seems like $7$ significant digits is all that's needed, at least in this specific training setting.

To emphasize this further, I ran $10$ trainings with the gradient rounded to $8$ and $16$ digits respectively.
Each run has a different random seed, but the equivalent runs with different significant digits have the same random seed,
for maximal comparability.
Here is the average of ten loss-curves for both settings:

![Validation loss over training tokens; average of 10 trainings; default model size; gradient rounded to 8 and 16 digits](results/images/val_loss_vs_token_from_percentage_0.3_depth_8_width_384_num_heads_1_digits_8_to_16_10_tries.png)

There seems to be no deviation between these two that cannot be explained by random noise.

The main takeaway from this section is that lowering the number of significant digits in the gradient
degrades model performance, or at best leaves it constant&mdash;as expected.

### L2 distance to losses of M_16

To see trends in both the model size and significant digits, I will plot the L2-distance between the validation
loss curves of a model (with the gradient rounded to different significant digits) to $M_{16}$, for different model sizes.

Here is the plot:

![L2 distance of loss curves to corresponding model with gradient rounded to 16 digits](results/images/l2_dist_val_loss_14_modelsizes_16_max_digits_0_min_digits.png)

This plot is dominated by $1$ to $4$ significant digits, so let's talk about those four.

I notice two things:

1. The performance degradation with decresing significant digits in the gradient is super-linear. This is similar to the performance degradation experienced by decreasing model size. (Usually, we look at this the other way around: the performance improvement from more digits / larger model size is sub-linear. That is equivalent.) This is some evidence for the explanation I've offered above.
2. With both increasing width and depth, the difference in performance increases. I believe that this is because the larger models have a much lower loss at $16$ significant gradient digits than the smaller ones, but almost all performance improvements are reflected in the L2-distance, because the models $M_{1}$-$M_{14}$ learn very little, regardless of size.

Let's now look at the same plot, but with only $7$ or more gradient digits, so that we can make out more details:

![L2 distance of loss curves to corresponding model with gradient rounded to 16 digits, 7-15 digits rounding](results/images/l2_dist_val_loss_14_modelsizes_16_max_digits_7_min_digits.png)

Some observations:

1. The L2-distance decreases with increasing scale, the opposite from before. Since the effect I spoke of above (of the model with the least gradient rounding benefitting most from scale) doesn't apply anymore, this must be another effect being revealed.
2. Looking closely: increasing depth decreases distance consistently, but the same cannot be said for width: at low depth, width decreases the distance, but at high depth it increases it.
3. The number of significant digits in the gradient is apparently irrelevant, as long as it exceeds $6$.

What is the reason for points $1$ through $3$?

My first thought was that training stability might increase with depth, which would mean less noise to increase the L2-distance between training runs.
To quantify this statement, I calculated the fast fourier transform (fft) of each training run's validation loss curve,
and calculated the mean frequency. The higher the frequency, the more spiky the loss curve.
Here are the mean frequencies:

![alt text](results/images/fft_mean_val_loss_14_modelsizes_10_digits_7_min_digits.png)

(I did not write the values into the boxes because I would have had to include too many significant digits.)

Clearly, the loss-curve frequency and the L2-distance to the $16$-gradient-digit loss curve are closely related.
However, the stability of training falls with increased model size.
This is the exact inverse of what I expected, which means that I have no real explanation for the effects of model scale on the L2 distance.

### L2 distance to losses of M_max

Let's do the same as above, but instead of comparing to $M_{16}$ at a given size,
we compare to $M_{max}$ at a given number of significant gradient digits.

Here is that plot:

![L2 distance of loss curve to that of largest model, at every number of gradient digits](results/images/l2_dist_val_loss_to_largest_model_13_modelsizes_16_digits_91137616_max_param_nums.png)

I can see two trends here:

1. The larger the model, the closer its loss curve is to that of the largest model. That is expected.
2. The difference between the loss curves tends to be greater for models trained on few significant gradient digits than for ones trained with many digits.

Let's again look at only the training runs with $6$ or more significant gradient digits:

![L2 distance of loss curve to that of largest model, at every number of gradient digits](results/images/l2_dist_val_loss_to_largest_model_13_modelsizes_11_digits_91137616_max_param_nums.png)

The L2-distance to the largest model is much more "stable" here between different training runs.

Three points:

1. As expected, the training dynamics of larger models are more similar to those of $M_{max}$ than those of smaller models.
2. The depth seems to be very significant; the model with $16$ layers of width $384$ has a loss curve closer to $M_{max}$ than the one with $4$ layers and a width of $576$, despite the latter having more parameters.
3. The above trends look very similar to the trends seen when comparing to $M_{16}$: as the models get larger, the rounding of the gradient has less of an effect, and their performance also gets closer to that of the largest model. There is something there, though I'm not fully sure what it is.

## Summary

Results:

- Rounding the gradient to $7$ or more digits doesn't visibly degrade performance.
- Rounding it to $6$ or fewer significant digits degrades performance.
- That degradation increases super-linearly with a lowering number of significant digits.
- This is (weak) evidence for the hypothesis that scaling laws in the number of parameters occur due to
    increased continuity between behaviors with increased scale, which makes the local search performed by SGD easier.

Open question:

- Is there really no performance degradation for models trained with gradients that were rounded to $7$ or more significant digits,
    or does it only look like that because the differences are so small?

To find the answer, running the same experiments again, but for more epochs and at least $20$ training runs each
would be informative.

Again, this was done on a whim without prior research; if you have any feedback or information, feel free to send it to @omouamoua on X/Twitter.

## Acknowledgements

This repo is based on my [hlb-gpt-cli](https://github.com/snimu/hlb-gpt-cli),
which in turn is based on [Fern](https://github.com/tysam-code)'s [hlb-gpt](https://github.com/tysam-code/hlb-gpt).

Here is the citation to that package:

```
cff-version: 1.2.0
message: "Citations would be appreciated if you end up using this tool! I currently go by Fern, no last name given."
authors:
  given-names: "Fern"
title: "hlb-gpt"
version: 0.4.1
date-released: 2023-03-05
url: "https://github.com/tysam-code/hlb-gpt"
```

If you make use of these results, please cite both this package and Fern's.
