**Background and Motivation**
Stochastic volatility (SV) models capture the fact that financial return variances evolve over time, addressing shortcomings of constant-variance (GARCH) frameworks.  In a basic SV model,

$$
\begin{aligned}
y_t &= \exp\bigl(h_t/2\bigr)\,\varepsilon_t,\quad \varepsilon_t\sim N(0,1),\\
h_{t+1} &= \mu + \phi\,(h_t - \mu) + \sigma\,\eta_{t+1},\quad \eta_{t+1}\sim N(0,1),
\end{aligned}
$$

where $h_t$ is the latent log‐variance.  Direct likelihood inference requires integrating over the latent $h_{1:T}$, leading to a $T$-dimensional integral that is analytically intractable.  Kim, Shephard, and Chib (1998) developed an MCMC scheme that turns this challenge into a sequence of simple updates by cleverly “Gaussianizing” the measurement equation ([Olin Apps][1]).

---

## 1. Mixture-of-Normals Approximation

The key innovation is to re-express the measurement equation in a form amenable to Gaussian state-space methods.  Observe

$$
\log(y_t^2)\;=\;h_t\;+\;\underbrace{\log(\varepsilon_t^2)}_{\omega_t},
$$

where $\omega_t$ follows a non-Gaussian “log–chi-squared” distribution.  Kim et al. approximate the density of $\omega_t$ by a finite mixture of $K$ normal components:

$$
p(\omega_t) \approx \sum_{i=1}^K \pi_i\,\mathcal{N}(\omega_t\mid m_i,v_i),
$$

with pre-tabulated weights $\{\pi_i\}$, means $\{m_i\}$, and variances $\{v_i\}$ (commonly $K=7$).  This transforms the problematic measurement noise into a tractable Gaussian mixture ([Olin Apps][1]).

---

## 2. Model Augmentation via Latent Indicators

Introduce latent indicators $r_t\in\{1,\dots,K\}$ such that, conditional on $r_t=i$,

$$
\log(y_t^2)\;=\;h_t \;+\; m_i \;+\;\nu_{t},\quad
\nu_t\sim N(0,v_i).
$$

Thus the augmented model is

$$
\begin{cases}
\log(y_t^2)\mid h_t,r_t=i &\sim N\bigl(h_t + m_i,\,v_i\bigr),\\
h_{t+1}\mid h_t &\sim N\bigl(\mu + \phi(h_t-\mu),\,\sigma^2\bigr).
\end{cases}
$$

This is a **linear Gaussian state-space model**, conditional on the full path $r_{1:T}$ ([Wikipedia][2]).

---

## 3. The Gibbs Sampling Algorithm

With the mixture augmentation, Kim et al. employ a three-block Gibbs sampler that cycles through:

1. **Sampling the mixture indicators $r_{1:T}$**
2. **Sampling the latent volatilities $h_{1:T}$ in one block**
3. **Sampling the SV parameters $(\mu,\phi,\sigma^2)$**

Each block exploits conditional conjugacy or standard Gaussian filtering/smoothing.

### Step 1: Sample Mixture Indicators $r_t$

For each $t$, draw

$$
r_t \;\sim\; \Pr(r_t=i \mid y_t,\,h_t)\;\propto\;\pi_i\;\exp\!\Bigl(-\tfrac{1}{2v_i}\bigl[\log(y_t^2)-h_t-m_i\bigr]^2\Bigr).
$$

Since the prior $\pi_i$ and the Gaussian likelihood are known, computing these $K$ weights and sampling $r_t$ is straightforward ([Olin Apps][1]).

### Step 2: Sample Latent Volatilities $h_{1:T}$

Conditional on $\{r_t\}$ and current $(\mu,\phi,\sigma^2)$, the model is a **Gaussian** state-space system.  Kim et al. apply the **Forward Filtering Backward Sampling (FFBS)** algorithm (Carter & Kohn, 1994; Shephard, 1994) to jointly sample the entire vector $h_{1:T}$ from its posterior.  FFBS proceeds by:

* **Forward pass**: run the Kalman filter to compute filtered means and variances for each $t$.
* **Backward pass**: sample $h_T$ from its filtered distribution, then recursively sample $h_{t}$ given $h_{t+1}$ and the filter output.
  This block update greatly improves mixing compared to one-at-a-time draws ([Martin Sewell Finance][3]).

### Step 3: Sample SV Parameters $(\mu,\phi,\sigma^2)$

Given $h_{1:T}$, the AR(1) evolution
$\;h_{t+1} = \mu + \phi(h_t-\mu) + \sigma\,\eta_{t+1}$\\
is just a linear regression with Gaussian errors.  With conjugate priors (e.g.\ Normal for $\mu,\phi$, Inverse-Gamma for $\sigma^2$), one obtains closed-form posterior conditionals:

* $\mu$ and $\phi$ from a bivariate normal update
* $\sigma^2$ from an Inverse-Gamma update
  Sampling these in one block maintains good efficiency ([Wikipedia][2]).

---

## 4. Importance Reweighting (Optional)

Because the mixture‐of‐normals is only an approximation, Kim et al. propose **importance reweighting** to correct any bias.  Each full draw $(h,r,\mu,\phi,\sigma^2)$ receives a weight proportional to the ratio of the true likelihood to the approximation.  These weights can be used to compute marginal likelihoods or properly weighted posterior summaries ([Olin Apps][1]).

---

## 5. Logical Advantages of the Method

* **Tractability**: The awkward non-Gaussian measurement noise is replaced by a Gaussian mixture, unlocking powerful Kalman‐based tools.
* **Block Updates**: Multi‐move sampling of $h_{1:T}$ via FFBS avoids the slow mixing of single‐site updates.
* **Conjugacy**: Conditional Gaussianity yields straightforward draws for both latent states and parameters.
* **Efficiency**: Empirical results in Kim et al. (1998) show rapid convergence and low autocorrelations in the MCMC output compared to prior SV sampling schemes ([Olin Apps][1], [economics.uci.edu][4]).

---

### Summary of the Step-by-Step Procedure

1. **Initialize** $(h_{1:T},r_{1:T},\mu,\phi,\sigma^2)$.
2. **Repeat** for each MCMC iteration:

   * **Sample** $r_t$ for $t=1,\dots,T$.
   * **Sample** $h_{1:T}$ jointly via FFBS.
   * **Sample** $(\mu,\phi,\sigma^2)$ from their posteriors.
   * *(Optional)* compute importance weights.
3. **Aggregate** posterior draws (weighted if reweighting is used) for inference on volatility paths and parameters.

This landmark methodology by Kim, Shephard, and Chib revolutionized SV estimation by marrying mixture approximations with efficient state-space sampling, and it remains the cornerstone of Bayesian SV analysis today.

[1]: https://apps.olin.wustl.edu/faculty/chib/papers/KimShephardChib98.pdf?utm_source=chatgpt.com "[PDF] Stochastic Volatility : Likelihood Inference and Comparison with ..."
[2]: https://en.wikipedia.org/wiki/Siddhartha_Chib?utm_source=chatgpt.com "Siddhartha Chib"
[3]: https://finance.martinsewell.com/stylized-facts/volatility/KimShephardChib1998.pdf?utm_source=chatgpt.com "[PDF] Stochastic Volatility: Likelihood Inference and Comparison ... - Finance"
[4]: https://www.economics.uci.edu/files/docs/colloqpapers/w06/chib.pdf?utm_source=chatgpt.com "[PDF] Stochastic volatility with leverage: fast and efficient likelihood inference"
