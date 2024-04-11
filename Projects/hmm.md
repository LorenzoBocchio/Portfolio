---
layout: default
title: HMMs for longitudinal data
description: Modeling longitudinal data with Hidden Markov Models (HMMs) to improve parameter estimation
img: assets/img/project_hmm/logo.jpg
importance: 3
category: Theoretical Insights
---


# Table of Contents
- [Introduction](#introduction)
- [Pooling](#pooling)
- [Models with continuous-valued random effects](#models-with-continuous-valued-random-effects)
- [Models with discrete-valued random effects](#models-with-discrete-valued-random-effects)
- [Discussion](#discussion)

<h2>Introduction</h2>
<a name="introduction"></a>
<ul>
  <li>Longitudinal data: K time series made of the same type of observations on K subjects. Used to model, for example, the status of a disease in each of the K individuals or the precipitation amount at each K site.</li>
  <li>Component Series: assume independence between each one of the K component series. The joint likelihood is just the product of the K individual ones.</li>
  <li>Usually, Component Series are not long enough to be fitted on their own. What's the solution?
    <ul>
      <li>Assume that some parameters are the same for every subject: pooling, which is advantageous even when the component series are long enough to be fitted singularly.</li>
      <li>Use Covariate Information. These covariates can be subject-specific or time-varying, expressing the parameters of the Transition probability matrix or of the state-dependent distributions by functions of the covariates.</li>
      <li>Models with random effects, also known as mixed Hidden Markov Models (HMMs): reducing the total number of parameters by regarding some of the parameters of the transition probability matrix or of the state-dependent distributions as realizations of random variables.</li>
    </ul>
  </li>
  <li>We consider sets {x<sub>tk</sub> : t = 1,...,T , K = 1,..., K}, representing counts, and fit a stationary two-state Poisson–HMM to each of the K. Parameters for the k-th subject include lambda<sub>1</sub><sup>(k)</sup>, lambda<sub>2</sub><sup>(k)</sup> and Gamma<sup>(k)</sup> representing the parameters for the k-th subject.</li>
</ul>

<h2>Pooling</h2>
<a name="pooling"></a>
<ul>
  <li>Pooling: reducing the total number of (free) parameters to be estimated, leading to a reduction in the standard error (s.e.) of the estimators.</li>
  <li>No Pooling: estimate 4 parameters for each component series: 4K parameters in total. Estimates obtained with no pooling provide a useful starting point, distinguishing the parameters that are approximately constant across subjects.</li>
  <li>Complete Pooling: estimate only 4 parameters, it provides a baseline model, useful for showing the benefit of applying alternative models with partial pooling.</li>
  <li>Partial Pooling: either pooling the transition probability matrix (t.p.m.) parameters or the state-dependent distributions' ones (2K+2 parameters).</li>
  <li>The Likelihood formula is simplified for web presentation, noting that it involves the product of probabilities across all K subjects and all time points.</li>
</ul>

<h2>Models with continuous-valued random effects</h2>
<a name="models-with-continuous-valued-random-effects"></a>
<ul>
  <li>Assume that some model parameters are continuous-valued random effects, independently drawn from a distribution that is common to all component series, with one realization for each series.</li>
  <li>For the case with the t.p.m parameters pooled, but not the ones of the state-dependent distributions, considering non-pooled parameters as fixed, we'd have 2K+2 parameters in total.</li>
  <li>The likelihood of this model is more demanding to compute, requiring integration over the distributions of the random effects. Specific distributions for the parameters of the state-dependent process might include, for example, the gamma distribution for the Poisson mean in each state.</li>
</ul>

<h2>Models with discrete-valued random effects</h2>
<a name="models-with-discrete-valued-random-effects"></a>
<ul>
  <li>This approach avoids integration and often reduces the computational efforts.</li>
  <li>Consider the case of partial pooling where lambda<sub>1</sub> and lambda<sub>2</sub> are assumed to be discrete-valued random variables taking on finitely many values.</li>
  <li>The likelihood involves summations rather than integrations, simplifying the computation.</li>
</ul>

<h2>Discussion</h2>
<a name="discussion"></a>
<ul>
  <li>A defining property of longitudinal data is that observations of the same type are available for each of the subjects. This leads to the opportunity to identify how subjects differ and how are similar.</li>
  <li>Pooling the parameters of the state-dependent distributions allows for variability in the state-switching dynamics of subjects, including the times spent in the different states. Alternatively, pooling the parameters of the t.p.m. allows for subject-specific variability within each state.</li>
  <li>Subject-specific covariates, if available, can explain some of the differences between subjects. Covariate information is easily incorporated into HMMs.</li>
  <li>The use of random effects is a convenient way to reduce the number of parameters that need to be estimated, especially when the number of subjects is large. However, incorporating random effects in HMMs is computationally very demanding.</li>
</ul>
