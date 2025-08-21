Let’s break it down the details of acquier, Polson and Rossi's (1994) Metropolis-within-Gibbs sampler in two parts: first how we choose priors for and then sample
$\omega=(\alpha,\delta,\sigma_v^2)$; and then how we sample each latent volatility $h_t$.

---

## 1. Conjugate Gibbs for $\omega=(\alpha,\delta,\sigma_v^2)$

We have the state‐equation

$$
\log h_t \;=\;\alpha \;+\;\delta\,\log h_{t-1}\;+\;v_t,\quad
v_t\sim N(0,\sigma_v^2),
$$

for $t=2,\dots,T$.  Writing this as a simple linear regression

$$
\underbrace{\begin{pmatrix}\log h_2\\ \log h_3\\ \vdots\\ \log h_T\end{pmatrix}}_{y}
\;=\;
\underbrace{\begin{pmatrix}1 & \log h_1\\1 & \log h_2\\ \vdots & \vdots\\1 & \log h_{T-1}\end{pmatrix}}_{X}
\begin{pmatrix}\alpha\\\delta\end{pmatrix}
\;+\;\underbrace{\begin{pmatrix}v_2\\v_3\\\vdots\\v_T\end{pmatrix}}_{v},
$$

we put the usual Normal–Inverse‐Gamma conjugate prior:

1. $\;(\alpha,\delta)\mid\sigma_v^2\;\sim\;N\big(m_0,\;\sigma_v^2\,V_0\big)$
2. $\;\sigma_v^2\;\sim\;\text{Inv‐Gamma}\big(\tfrac{\nu_0}{2},\tfrac{S_0}{2}\big)$

Here $m_0$ (a 2–vector) and $V_0$ (a $2\times2$ matrix), plus $\nu_0,S_0$, are hyper‐parameters you choose to reflect your prior belief (e.g.\ vague by letting $V_0^{-1}$ be very small, $\nu_0$ small, etc.).

Given the observed “data” $\{\,\log h_t\}$, the **posterior** is again Normal–Inverse‐Gamma:

1. **Posterior of** $\sigma_v^2$:

   $$
     \sigma_v^2 \;\big|\;h\;\sim\;\text{Inv‐Gamma}\Big(\tfrac{\nu_0 + (T-1)}{2},\;\tfrac{S_0 + \mathrm{SSR}}{2}\Big),
   $$

   where $\mathrm{SSR} = (y - X\,\hat\beta)^\top(y - X\,\hat\beta)$ is the sum of squared residuals from the regression, and
2. **Posterior of** $(\alpha,\delta)$:

   $$
     (\alpha,\delta) \;\big|\;h,\;\sigma_v^2
     \;\sim\;
     N\big(m_n,\;\sigma_v^2\,V_n\big),
   $$

   with

   $$
   V_n^{-1} = V_0^{-1} + X^\top X,
   \quad
   m_n = V_n\big(V_0^{-1}m_0 + X^\top y\big).
   $$

**Sampling step**:

* First draw $\sigma_v^2$ from its Inv‐Gamma posterior.
* Then draw $(\alpha,\delta)$ from the 2–variate Normal above.

Because these are closed‐form, this is a **pure Gibbs** update.

---

## 2. Metropolis-within-Gibbs for the latent $h_t$

Once $\omega$ is updated, we turn to each $h_t$.  The full conditional for a single $h_t$ (holding all other $h_{s\neq t}$ and $\omega$ fixed) is

$$
p(h_t\mid y_t, h_{t-1},h_{t+1},\omega)
\;\propto\;
\underbrace{\,\frac{1}{\sqrt{h_t}}\exp\!\Big(-\frac{y_t^2}{2\,h_t}\Big)\!}_{\text{obs. density}}
\;\times\;
\underbrace{\frac{1}{h_t}\exp\!\Big(-\frac{(\log h_t - \mu_t)^2}{2\,V}\Big)\!}_{\text{“prior” from AR neighbors}},
$$

where

* $V = \displaystyle\frac{\sigma_v^2}{1+\delta^2}$,
* $\mu_t$ is the precision‐weighted mean of $\log h_{t-1}$ and $\log h_{t+1} - \alpha$ (the two AR links).

This density is **log‐concave** in $h_t$ but has no standard sampler.  Jacquier–Polson–Rossi proceed in two steps:

---

### 2.1 Build an Inverse‐Gamma proposal $q(h_t)$

1. **Moment‐match** the “Gaussian on $\log h_t$” factor by finding parameters $(\phi,\theta)$ so that

   $$
     q(h_t)\;\propto\;h_t^{-(\phi+1)}\exp\!\Big(-\frac{\theta}{h_t}\Big)
     \quad\text{(an Inv‐Gamma law)},
   $$

   matches the **first two moments** of the AR‐based part.
2. Multiply by the observation‐part’s kernel $h_t^{-1/2}e^{-y_t^2/(2h_t)}$ and re‐normalize—what you get is again an Inv‐Gamma.

Because this tailors $q$ to look like the true conditional, it is a **tight envelope**.

---

### 2.2 Two‐stage acceptance

1. **Rejection sampler**

   * Draw $h_t^*\sim q(h_t)$.
   * Accept with probability

     $$
       \alpha_{\rm rej} 
       = \min\Biggl\{\,\frac{p(h_t^*)}{c\,q(h_t^*)}\;,\;1\Biggr\},
     $$

     where $c$ is chosen so that $c\,q$ dominates $p$; they pick $c$ ≈1.1× the supremum of $p/q$.
   * If rejected, stay at the old $h_t$; otherwise move to $h_t^*$.

2. **Metropolis–Hastings correction**
   Even if the rejection sampler weren’t perfect, you then wrap one MH step around it to enforce detailed balance.  Explicitly the MH acceptance ratio is

   $$
     \alpha_{\rm MH}
     = \min\Bigl\{\,1,\,
       \frac{p(h_t^*)\,q(h_t^{\rm old})}{p(h_t^{\rm old})\,q(h_t^*)}
     \Bigr\}.
   $$

   In practice this second correction almost always accepts, so overall you get ≈80–90% acceptance.

---

## 3. Putting it all together (“Metropolis-within-Gibbs”)

On each full MCMC iteration:

1. **Gibbs‐draw** $\sigma_v^2$, then $(\alpha,\delta)$ from their exact posterior given the current $\{h_t\}$.
2. **For** $t = 1,\dots,T$: sample $h_t$ by the two‐stage proposal→reject→MH above, holding $\omega$ and the other $h$’s fixed.

Because each block is either an exact Gibbs block or a valid MH block targeting the true full conditional, the chain is guaranteed to converge to the joint posterior

$$
p\bigl(\omega,\,h_1,\dots,h_T\mid y_1,\dots,y_T\bigr).
$$

---

### Why this makes sense

* **Conjugate Gibbs** for $\omega$ is super‐efficient: no tuning, exact draws, and you exploit the linear‐regression structure in $\log h$.
* **Tailored Metropolis** for each $h_t$ exploits the fact that the only thing stopping an analytic draw is the messy $h_t^{-3/2}e^{-(\cdots)}$ form.  By fitting an Inv‐Gamma envelope to the smooth “AR” part, you get a cheap sampler with high acceptance.
* Splitting into two blocks keeps each update simple and low‐dimensional—this dramatically reduces autocorrelation compared to trying to update all $T$ latent states jointly.

Together, these ideas give an MCMC that mixes quickly even for long time series of stochastic volatility.

# Part 2 - more details on the latent variable part of MCMC
Let’s peel back each piece of that latent-$h_t$ step in detail.

---

## A. What is the “AR-based” part?

Recall the full conditional for one latent $h_t$ (dropping subscripts on $\omega$) is

$$
p(h_t\mid y_t,h_{t-1},h_{t+1})
\;\propto\;
\underbrace{\frac{1}{\sqrt{h_t}}
\exp\!\biggl(-\frac{y_t^2}{2\,h_t}\biggr)}_{\displaystyle\text{(I) observation kernel}}
\;\times\;
\underbrace{\frac{1}{\,h_t\,}
\exp\!\Bigl(-\frac{(\log h_t-\mu_t)^2}{2 V}\Bigr)}_{\displaystyle\text{(II) AR‐prior kernel}},
$$

where

* $\mu_t$ and $V=\sigma_v^2/(1+\delta^2)$ come from the two AR links $\log h_t\mid\log h_{t-1},\log h_{t+1}$.
* (I) is itself an Inverse‐Gamma–type kernel on $h_t$ (shape $½$, scale $y_t^2/2$).
* (II) is the log-normal kernel induced by the AR(1) on $\log h_t$.

We **cannot** sample directly from the product of a log-normal and an IG kernel.

---

## B. Approximating the AR-part by an Inverse‐Gamma

1. **Moments of the AR-part**
   The AR-piece (II) says

   $$
     \log h_t\;\sim\;N(\mu_t,\;V)
     \;\Longrightarrow\;
     h_t\;\sim\;\text{Lognormal}(\mu_t,V).
   $$

   A lognormal has

   $$
     m \;=\; \mathbb{E}[h_t]
     = e^{\mu_t + V/2},
     \quad
     s^2 \;=\; \operatorname{Var}(h_t)
     = \Bigl(e^V - 1\Bigr)\,e^{2\mu_t + V}\,.
   $$

2. **Match to an Inverse‐Gamma**
   An $\mathrm{Inv}\textrm{-}\Gamma(\phi,\theta)$ has

   $$
     \mathbb{E}[h] = \frac{\theta}{\phi - 1},\quad
     \operatorname{Var}(h) = \frac{\theta^2}{(\phi - 1)^2(\phi - 2)}.
   $$

   We **solve** for $\phi_{\rm prior},\theta_{\rm prior}$ so that

   $$
     \frac{\theta_{\rm prior}}{\phi_{\rm prior}-1}=m,
     \quad
     \frac{\theta_{\rm prior}^2}{(\phi_{\rm prior}-1)^2(\phi_{\rm prior}-2)}=s^2.
   $$

   In this way the IG “envelope” has exactly the same mean and variance as the lognormal AR-part.  That makes it a very **tight** cover of (II).

---

## C. Building the proposal $q(h_t)$

Because the observation kernel (I) is already

$$
h_t^{-1/2}\,\exp\!\Bigl(-\frac{y_t^2}{2h_t}\Bigr)
\;\propto\;
\mathrm{Inv}\text{-}\Gamma\Bigl(\tfrac12,\;\tfrac{y_t^2}{2}\Bigr),
$$

we take our matched $ \mathrm{Inv}\text{-}\Gamma(\phi_{\rm prior},\,\theta_{\rm prior})$ and **multiply** it by (I).  The product of two IG–kernels is again an IG–kernel, now with

$$
\begin{aligned}
  \phi_q &= \phi_{\rm prior} + \tfrac12,\\
  \theta_q &= \theta_{\rm prior} + \tfrac{y_t^2}{2}.
\end{aligned}
$$

Thus

$$
q(h_t)\;=\;\mathrm{Inv}\text{-}\Gamma\bigl(\phi_q,\theta_q\bigr).
$$

> **Why multiply?**
> We want $q$ to approximate the *entire* full conditional, not just the AR-part.  By folding in (I) we get a single IG that’s close in shape to
> $\text{(I)}\times\text{(II)}$.

---

## D. Two‐stage sampling

### 1. Rejection‐sampling against the envelope

We choose a constant $c\ge\sup_h\bigl[p(h)/q(h)\bigr]$ (in practice $c\approx1.1\times$ the supremum).  Then:

* Draw $h^*\sim q(h)$.
* With probability

  $$
    \alpha_{\rm rej}
    = \min\Bigl\{\;\tfrac{p(h^*)}{c\,q(h^*)}\;,\;1\Bigr\},
  $$

  **accept** $h^*$ *as if* it were from the true conditional.  Otherwise **reject** and stay at the old $h_t$.

This alone would give exact draws *if* $c\,q$ perfectly envelops $p$.  But since we found $q$ by moment-matching, there may be tiny pockets where $p(h)>c\,q(h)$.

### 2. MH‐correction to “fix” any envelope slip

To guarantee we target exactly
$\,p(h_t\mid\cdots)$, we treat the result of the rejection sampler as a *proposal* in a Metropolis–Hastings step with target $p$.  Concretely:

* If the rejection stage accepted $h^*$, we then compute

  $$
    \alpha_{\rm MH}
    = \min\!\Bigl\{1,\;\frac{p(h^*)\,q(h_{\rm old})}
                          {p(h_{\rm old})\,q(h^*)}\Bigr\},
  $$

  and accept or reject accordingly.

* If the rejection stage already rejected $h^*$, we remain at $h_{\rm old}$.

Because the rejection step already catches ≈ 80–90% of the mass correctly, this MH‐step almost **always** accepts — but it insures **detailed balance** and exact targeting.

---

## E. Why not just single‐stage MH?

* **Pure MH** with proposal $q$ would work, but its acceptance would be
  $\min\{1,\,p(h^*)\,q(h_{\rm old})/(p(h_{\rm old})\,q(h^*))\}$,
  which tends to be lower (≈ 50–60%) and leads to higher autocorrelation.
* The **two‐stage** design gets you:

  * A **fast rejection sampler** that directly draws \~80% of samples exactly from $p$.
  * A tiny MH correction to patch up the ≲ 20% of cases where the envelope wasn’t perfect.
* Net effect: **much higher effective sample size** per computation.

---

### Summary of one $h_t$ update

1. **Form** the AR‐neighbor moments $(m,s^2)$.
2. **Solve** for IG-parameters $(\phi_{\rm prior},\theta_{\rm prior})$ matching $(m,s^2)$.
3. **Define** proposal

   $$
   q(h)\;=\;\mathrm{Inv}\text{-}\Gamma\bigl(\phi_{\rm prior}+\tfrac12,\,
                                             \theta_{\rm prior}+\tfrac{y_t^2}{2}\bigr).
   $$
4. **Rejection step** with envelope constant $c$.
5. **MH step** on the accepted draw to enforce exactness.

That two‐stage sampler is what lets you cycle through $t=1,\dots,T$ very efficiently, with very little autocorrelation, yet still target the *exact* full conditional $p(h_t\mid\cdots)$.
