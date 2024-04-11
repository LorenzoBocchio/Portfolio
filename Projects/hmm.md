---
layout: default
title: HMMs for longitudinal data
description: Modeling longitudinal data with Hidden Markov Models (HMMs) to improve parameter estimation
img: assets/img/project_hmm/logo.jpg
importance: 3
category: Theoretical Insights
---
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
    <h1>Table of Contents</h1>
    <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#pooling">Pooling</a></li>
        <li><a href="#models-with-continuous-valued-random-effects">Models with continuous-valued random effects</a></li>
        <li><a href="#models-with-discrete-valued-random-effects">Models with discrete-valued random effects</a></li>
        <li><a href="#discussion">Discussion</a></li>
    </ul>

    <h2 id="introduction">Introduction</h2>
    <p>- Longitudinal data: K time series made of the same type of observations on K subjects. Used to model, for example, the status of a disease in each of the K individuals or the precipitation amount at each K site.</p>
    <p>- Component Series: assume independence between each one of the K component series. The joint likelihood is just the product of the K individual ones.</p>
    <p>- Usually, Component Series are not long enough to be fitted on their own. What's the solution?</p>
    <ul>
        <li>Assume that some parameters are the same for every subject: pooling, which is advantageous even when the component series are long enough to be fitted singularly.</li>
        <li>Use Covariate Information. These covariates can be subject-specific or time-varying, expressing the parameters of the Transition probability matrix or of the state-dependent distributions by functions of the covariates.</li>
        <li>Models with random effects, also known as mixed HMMs: reducing the total number of parameters by regarding some of the parameters of the t.p.m. or of the state-dependent distributions as realizations of random variables.</li>
    </ul>
    <p>We consider \( \{x_{tk} : t = 1,\ldots,T , K = 1,\ldots, K \} \), representing counts, and fit a stationary two-state Poisson–HMM to each of the \( K \). Let \( \lambda_1^{(k)} \), \( \lambda_2^{(k)} \) and \( \Gamma^{(k)} = (\gamma_{ij}^{(k)}) \) represent the parameters for the \( k \)-th subject.</p>

    <h2 id="pooling">Pooling</h2>
    <p>- Pooling: reducing the total number of (free) parameters to be estimated, leading to a reduction in the s.e. of the estimators.</p>
    <p>- No Pooling: estimate 4 parameters for each component series: 4K parameters in total. Estimates obtained with no pooling provide a useful starting point, distinguishing the parameters that are approximately constant across subjects.</p>
    <p>- Complete Pooling: estimate only 4 parameters, it provides a baseline model, useful for showing the benefit of applying alternative models with partial pooling.</p>
    <p>- Partial Pooling: either pooling the t.p.m parameters or the state-dependent distributions' ones (2K+2 parameters).</p>
    <p>- The Likelihood is the same, we only remove the \( k \) subscripts to the pooled parameters in each case.</p>
    <p>\[ L = \prod_{k = 1}^{K}\delta^{(k)}P^{(k)}(x_{1k})\Gamma^{(k)}P^{(k)}(x_{2k})\Gamma^{(k)}\ldots\Gamma^{(k)}P^{(k)}(x_{Tk})1' \]</p>

    <h2 id="models-with-continuous-valued-random-effects">Models with continuous-valued random effects</h2>
    <p>- Assume that some model parameters are continuous-valued random effects, independently drawn from a distribution that is common to all component series, with one realization for each series.</p>
    <p>- As an example, let's consider the partial pooling case with the t.p.m parameters pooled, but not the ones of the state-dependent distributions. Considering non-pooled parameters as fixed, we'd have 2K+2 parameters in total.</p>
    <p>- If instead we consider \( \lambda_1^{(k)} \), \( \lambda_2^{(k)} \), for \( k = 1,\ldots, K \) are regarded as random effects:</p>
    <ul>
        <li>Specify distributions for the parameters of the state-dependent process, e.g., \( \lambda_1^{(k)} \sim Ga(\mu_1, \sigma_1) \) and independently \( \lambda_2^{(k)} \sim Ga(\mu_2, \sigma_2) \)</li>
        <li>We need to estimate only six quantities: \( \gamma_{12}, \gamma_{21}, \mu_1, \sigma_1, \mu_2, \sigma_2 \).</li>
    </ul>
    <p>- The likelihood of this model is much more demanding to compute:</p>
    <p>\[ L = \prod_{k = 1}^{K} \int_{0}^{\infty} \int_{0}^{\infty}  \delta P(x_{1k})\Gamma P(x_{2k})\Gamma\ldots\Gamma 
    P (x_{Tk})1' \cdot f(\lambda_1, \mu_1, \sigma_1 ) f(\lambda_2, \mu_2, \sigma_2 ) d \lambda_1 d \lambda_2 \]</p>
    <p>\( f(\lambda_1, \mu_1, \sigma_1 ) \) is the p.d.f. of the gamma distribution for the Poisson mean in state 1, and \( f(\lambda_2, \mu_2, \sigma_2 ) \) for that in state 2.</p>
    <p>- One can also specify bivariate distributions, allowing for possible dependence between random effects, adding one additional parameter, a correlation coefficient.</p>
    <p>- A numerical or EM-based maximization of the likelihood is only feasible when there are no more than 2 random effects. Simulation-based methods, such as Monte Carlo EM or MCMC, seem preferable in the other cases. A computationally less intensive approach uses discrete distributions for the random effects.</p>

    <h2 id="models-with-discrete-valued-random-effects">Models with discrete-valued random effects</h2>
    <p>- It avoids integration and often reduces the computational efforts.</p>
    <p>- Consider the case of partial pooling: \( \lambda^{(k)}_1 \) and \( \lambda^{(k)}_2 \) are assumed to be discrete-valued random variables taking on finitely many values. Specifically, suppose that, for \( k = 1, \ldots , K \) and \( i = 1,2 \).</p>
    <p>- The total number of parameters is \( 2q_1 + 2q_2 +2 \).</p>
    <p>- The likelihood involves summations, rather than integrations:</p>
    <p>\[ \prod_{k = 1}^{K} \sum_{j_1 = 1}^{q_1}\sum_{j_2 = 1}^{q_2}  \delta P(x_{1k})\Gamma P(x_{2k})\Gamma\ldots\Gamma P (x_{Tk})1'\pi_{j_1 1}\pi_{j_2 2} \]</p>
    <p>- The use of discrete random effects also allows for a simple way to model dependence between multiple random effects. One could specify a single bivariate distribution.</p>

    <h2 id="discussion">Discussion</h2>
    <p>- A defining property of longitudinal data is that observations of the same type are available for each of the subjects. This leads to the opportunity to identify how subjects differ and how are similar.</p>
    <p>- Pooling the parameters of the state-dependent distributions allows for variability in the state-switching dynamics of subjects, including the times spent in the different states. Alternatively, pooling the parameters of the t.p.m. allows for subject-specific variability within each state.</p>
    <p>- Subject-specific covariates, if available, can explain some of the differences between subjects. Covariate information is easily incorporated into HMMs.</p>
    <p>- The use of random effects is a convenient way to reduce the number of parameters that need to be estimated, especially when the number of subjects is large. There is a disadvantage to incorporating random effects in HMMs, which is that their implementation is computationally very demanding. The computational effort can be reduced by using discrete-valued rather than continuous-valued random effects.</p>
</body>
</html>

