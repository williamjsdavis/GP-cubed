## Hacking Non-Stationary Gaussian Processes

This repository is related to the following blog post:

[Hacking Non-Stationary Gaussian Processes](https://williamdavis.name/blog/hacking-non-stationary-gaussian-processes/)

The repository contains the code used to produce the blog post.

## Installation

Use the poetry `pyproject.toml` file to install the dependencies.

```bash
poetry install
```

## Blog Post

(The main text here is copied directly from the blog post.)

### Introduction

Gaussian processes (GPs) are a powerful tool for modeling and quantifying uncertainty over functions that are smooth and continuous. 
In fact, Gaussian process regression is one of the only methods for Bayesian inference that allows you to operate in infinite-dimensional spaces, without either: resorting to approximate inference methods like RTO-TKO [1], or truncating the problem *ab initio* and using typical Bayesian sampling methods (i.e., MCMC) [2].
So, GPs are great! In my work at Terra AI I use them all the time for tasks like modeling prior distributions of subsurface geologic properties (or geo-priors, as I call them), or modeling uncertain cost landscapes in Bayesian optimization.
It's also now easier than ever to get GP models up and running in production settings, thanks to the development of GP libraries like GPyTorch [3], GPyFlow [4], BoTorch [5], as well as the rich Julia ecosystem for GP modeling [6].

An important part of a GP is the covariance function, also known as the kernel, which encodes how pairs of points in the domain are related to each other. Kernel design is a key part of GP modeling: it allows you to encode prior beliefs about the function you're modeling, while also allowing you to flexibly fit the data [7].
However, many out-of-the box GP libraries are limited to stationary kernels. These are kernels that have the same properties everywhere in their domain. This is a limitation when modeling data that has different variability over different regions of interest.

Take, for example, the problem of modeling a sedimentary system that has different variability over different regions of interest. Perhaps it has large, lateral, parallel layers interspersed with smaller-scale, channelized features. A stationary kernel would be unable to model this, as it assumes that variability is similar everywhere in the domain.

(Plot: https://upload.wikimedia.org/wikipedia/commons/2/24/Channel-StellartonFm-CoalburnPit.JPG)
(Subtitle: Source: A sedimentary cross-section, showing lateral layers and a channelized feature. Image from Michael C. Rygel via Wikimedia Commons [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).)

Although the mathematics of GPs allows for spatially-varying kernels, one might be restricted to a stationary kernel by the library they are using. If this is the case, how can you "hack" non-stationarity into your GP? In this blog post, I'll show you three (well, two and a half) ways to do this. I'm also going to take this opportunity to announce the start of a new ongoing series of blog posts and a new GitHub repository: "The Gaussian Processes for GeoPhysics and Geology Playground.," or GP³ for short. These posts and examples (starting with my last post on full-matrix distances for GPs [8]) will be focused on novel ways to work with GPs, with a focus on the use of GPs in geophysics and geology. Let's get started!


### Approach 0: The Naive Approach

Let's start off with seeing how bad the problem is when we use a stationary kernel. I'm going to create a toy dataset that has different variability over different regions of interest, and then fit a GP to it using a stationary kernel. To see the code for this, you can check out the `hacking-non-stationary-GPs/0_naive_stationary_GP.ipynb` file. This model, as well as the rest of the models I'll show you in this post, are implemented using GPyTorch [3]. 

(Plot: 0_B_data.png)
(Subtitle: In the repo this is dataset B.)

Once I've trained my GP model on this data, I'll make predictions across the whole domain. In the spirit of "vibe-coding," the evaluation process here will be "vibe-verification," where I'll just see if it looks "right."

(Plot: 0_B_predictions.png)
(Subtitle: I'm plotting everything shown in the legend; in later plots I may hide certain elements.)

As we can see, the GP fits to the long-wavelength features, but shoots straight through the short-wavelength features. This is because the stationary kernel is not able to model the non-stationarity of the data. A different failure mode can occur in some (rarer) cases, shown below.

(Plot: 0_D_results_samples.png)
(Subtitle: I'm only plotting the posterior samples, not the mean or the uncertainty.)

Here, the model fits to the short-wavelength features, giving a very wiggly fit to smoother parts of the domain. This behavior can be understood by looking at the minimum loss for a range of fixed length-scales.

(Plot: 0_B_loss_length_scale.png)
(Subtitle: I fix the length-scale of the GP and train the other parameters (covariance and likelihood). The plotted data is the training loss, for a range of fixed length-scales.)

As we can see, there is a bimodal distribution of minimum loss, with a minimum at a length-scale that is too large for the short-wavelength features, and a minimum at a length-scale that is too small for the long-wavelength features. There is no way for this model to fit both parts of the variability well.

### Approach 1: Warped Inputs

One idea to hack non-stationarity into a GP is to "warp" the input space. This is the idea behind warped GPs, first introduced by Snelson et al. [9]. The idea is to transform the input space in a way that makes the problem stationary. It's key that this warping function is learnable, whether it's a neural network or a simple function like a polynomial. In my example in `hacking-non-stationary-GPs/2_segmented_GP.ipynb`, I defined the warping function as a simple neural network with a single hidden layer, and then jointly optimized the warping function and the GP parameters at the same time.

(Plot: 1_D_results_samples.png)
(Subtitle: I'm plotting the predictions in true-space in the top plot, and the predictions in warped-space in the bottom plot.)

Here the warping function has "squished" the left section of the domain, and "stretched" the middle section. This has the effect of making the variability the same wavelength throughout the domain. Looking at the predictions of the trained model, there isn't much high-frequency wiggling in the smooth parts of the domain: the model has fit both the long- and short-wavelength features well with its single-length-scale stationary kernel. The parts of the domain that have been warped can be visualized by plotting the gradient of the warping function.

(Plot: 1_D_results_scaling_factor.png)
(Subtitle: Bottom plot shows the scaling factor, i.e., the gradient, of the warping function.)

This learned warping function happens to be monotonically increasing, but in practice it may not be. It might be decreasing, or loop back on itself, depending on what fits the data best.

(Plot: 1_D_warping_function.png)
(Subtitle: The warping function, luckily, is monotonically increasing.)

If you wanted to do a proper Bayesian attempt at this, you can perform inference over the warping function as well, which has been done in the literature [10, 11].

### Approach 2: Segmented GPs

Another simple, but effective, approach to non-stationarity is to chop up the domain into different segments, and fit a GP to each segment. I couldn't find a concensus name for this method (as I'd bet this approach has been in use since the onset of GPs for data modeling), but I've seen it refered to as "segmented GPs," "partitioned GPs," or just "chunking." I've implemented this approach in `hacking-non-stationary-GPs/2_segmented_GP.ipynb`. I use `scikit-learn`'s `KMeans` clustering algorithm [12] to find the segments, although any segmentation method could be used.

(Plot: 2_B_results_samples.png)
(Subtitle: I'm plotting posterior samples and the data, which is color coded by the segment of the domain it belongs to.)

Here we can see that, although the model has fit each segment well, there are painful discontinuities at the segment boundaries. If your target application needs full-domain coherence, this is a no-go. On the plus side, as this method fits multiple GPs to smaller datasets, it is much more computationally efficient than other approaches. In fact, Tazi et al. [7] recommend this approach for large-scale datasets. GPs scale in compute and memory with O(N³) and O(N²) respectively, N being the number of training points. If you divide your domain into M segments, the complexities become O(N³/M³) and O(N²/M²). Additionally, training can be parallelized across the segments.

### Approach 3: Basis Function GPs

Ok so this is just a generalization of the segmented GP approach. Instead of partitioning the domain into discrete segments, you can instead can use any basis functions. The idea is to define a set of orthogonal basis (preferably overlapping) functions that sum to 1 at every point in the domain. Then, you train a GP for each basis on the data points in values of the domain where the basis function has non-zero weight. At evaluation time, you can sum the predictions of the GPs weighted by the basis function values to get the final prediction. I couldn't find an example of this approach in the literature, but I'm sure it's been done before. In my example in `hacking-non-stationary-GPs/3_basis_GP.ipynb`, I used a set of triangular basis functions.

(Plot: 3_B_results_samples.png)
(Subtitle: Bottom plot shows posterior samples and the data, and the top plot shows the basis functions in different colors.)

This model fits the data without the discontinuities of the segmented GP, giving a much smoother but variable fit. However this advantage is also a disadvantage: sharp changes in variability are smoothed out. The posterior samples near x=8 show this effect, where the basis function around x=8 learns the short-scale variability, and imposes this on the smoother area at x>8. Ideally, you could be able to learn the most effective basis functions for the problem at hand, but I'm not sure what format that would take.

### Other Approaches (That Didn't Work or I Didn't Try)

There were a few other approaches I researched and/or tried, so I'll briefly mention them here:

 - Spectral GPs, both with deltas [13] and mixtures [14]: Both approaches did not perform well for me; I think they're better suited to (semi-)periodic data.
 - Fully Bayesian GPs: I also tried using a fully Bayesian GP [15], with prior distributions over the length-scale parameter. Although this produced some posterior samples that fit short-scale variability, and some that fit long-scale variability, no single sample fit all of the variability well.
 - Hierarchical GPs: I found a cool paper by Lee et al. [16] pretty late into this project, and I wish I had seen it earlier. This uses a hierarchical sets of inducing points, and trains a GP for each level of the hierarchy to give a multi-scale model. I haven't tried this yet, but it's on my todo list.

### Conclusions

This was a fun dive into GPs for non-stationary data! This went more into the implementation and modeling choices than either the math or applications of GPs, but I hope you found it useful. 

I'd like to explore how these approaches perform on higher-dimensional data in the future, as well as real-world geological datasets. I'll be posting more about this in the future, so stay tuned!

Possible topics for future GP³ posts:

 - Hierarchical GPs for multi-scale modeling
 - GPs for gravity problems
 - Anandaroop Ray's approaches for GP EM inverse problems

### References

[1] Blatter et al., Uncertainty quantification for regularized inversion of electromagnetic geophysical data—Part I: motivation and theory, Geophysical Journal International, Volume 231, Issue 2, November 2022, Pages 1057–1074, https://doi.org/10.1093/gji/ggac241
[2] Sambridge and Mosegaard, Monte Carlo methods in geophysical inverse problems, Reviews of Geophysics, Volume 40, Issue 3, September 2002, Pages 3-1, https://doi.org/10.1029/2000RG000089
[3] https://gpytorch.readthedocs.io/en/latest/
[4] https://www.gpflow.org/
[5] https://botorch.org/
[6] https://github.com/JuliaGaussianProcesses
[7] Tazi et al., Beyond Intuition, a Framework for Applying GPs to Real-World Data. arXiv preprint arXiv:2307.03093. 2023 Jul 6. https://arxiv.org/abs/2307.03093
[8] https://posgeo.wordpress.com/2024/12/05/full-matrix-distances-for-gaussian-process-kernels/
[9] Snelson et al., Warped gaussian processes. Advances in neural information processing systems. 2003;16. https://proceedings.neurips.cc/paper/2003/hash/6b5754d737784b51ec5075c0dc437bf0-Abstract.html
[10] Lázaro-Gredilla, Bayesian warped Gaussian processes. Advances in Neural Information Processing Systems. 2012;25. https://proceedings.neurips.cc/paper_files/paper/2012/hash/d840cc5d906c3e9c84374c8919d2074e-Abstract.html
[11] Snoek et al., Input warping for Bayesian optimization of non-stationary functions. In International conference on machine learning 2014 Jun 18 (pp. 1674-1682). PMLR. https://proceedings.mlr.press/v32/snoek14.html
[12] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
[13] https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Spectral_Delta_GP_Regression.html
[14] https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Spectral_Mixture_GP_Regression.html
[15] https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_Fully_Bayesian.html
[16] Lee et al., Hierarchically-partitioned Gaussian process approximation. In Artificial Intelligence and Statistics 2017 Apr 10 (pp. 822-831). PMLR. https://proceedings.mlr.press/v54/lee17a.html