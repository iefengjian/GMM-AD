import math
import torch

class GMM_AD:
    """
    GMM_AD trained by EM on GPU (chunked).
    Each component k:
        x ~ N(mu_k, C_k),  C_k = W_k W_k^T + sigma2_k I
    where W_k: (D, d_latent), sigma2_k: scalar.

    """

    def __init__(
        self,
        K=32,
        d_latent=8,        # latent dimension per component (rank)
        iters=30,
        eps=1e-6,
        seed=0,
        x_chunk=65536,
        dtype=torch.float32,
        # ===== optional capacity controls =====
        topm: int = None,     # keep top-m responsibilities per sample (None disables)
        alpha: float = 1.0,   # Dirichlet concentration for pi
        prune_eps: float = 0.0,
        prune_every: int = 5,
        # ===== numerical safety =====
        sigma2_min: float = 1e-4,   # min isotropic noise
        eig_eps: float = 1e-6,      # clamp for eigenvalues
        # ===== scoring constants =====
        use_float32_accum: bool = True,  # accumulate stats in float32 (recommended)
    ):
        self.K = int(K)
        self.d_latent = int(d_latent)
        self.iters = int(iters)
        self.eps = float(eps)
        self.seed = int(seed)
        self.x_chunk = int(x_chunk)
        self.dtype = dtype

        self.topm = topm if topm is None else int(topm)
        self.alpha = float(alpha)
        self.prune_eps = float(prune_eps)
        self.prune_every = int(prune_every)

        self.sigma2_min = float(sigma2_min)
        self.eig_eps = float(eig_eps)
        self.use_float32_accum = bool(use_float32_accum)

        # params
        self.fitted = False
        self.log_pi = None     # (K,)
        self.mu = None         # (K,D)
        self.W = None          # (K,D,d_latent)
        self.log_sigma2 = None # (K,)
        self.alive = None      # (K,) bool

    # ---------------- utils ----------------

    @torch.no_grad()
    def _init_kmeanspp(self, X):
        """
        KMeans++ init for means only.
        X: (N,D) cuda
        """
        N, D = X.shape
        K = self.K
        g = torch.Generator(device=X.device)
        g.manual_seed(self.seed)

        mu = torch.empty((K, D), device=X.device, dtype=X.dtype)
        idx = torch.randint(0, N, (1,), generator=g, device=X.device)
        mu[0] = X[idx]

        d2 = torch.sum((X - mu[0])**2, dim=1)  # (N,)
        for k in range(1, K):
            probs = d2 / (d2.sum() + self.eps)
            idx = torch.multinomial(probs, 1, generator=g)
            mu[k] = X[idx]
            d2 = torch.minimum(d2, torch.sum((X - mu[k])**2, dim=1))
        return mu

    @torch.no_grad()
    def _apply_topm(self, r: torch.Tensor) -> torch.Tensor:
        # r: (B,K)
        if self.topm is None:
            return r
        K = r.shape[1]
        m = min(self.topm, K)
        if m >= K:
            return r
        vals, idx = torch.topk(r, k=m, dim=1, largest=True, sorted=False)
        r2 = torch.zeros_like(r)
        r2.scatter_(1, idx, vals)
        r2 = r2 / (r2.sum(dim=1, keepdim=True) + self.eps)
        return r2

    @torch.no_grad()
    def _update_pi_with_capacity(self, Nk: torch.Tensor) -> torch.Tensor:
        """
        Nk: (K,) float tensor
        """
        eps = self.eps
        if self.alpha == 1.0 and (self.prune_eps <= 0):
            pi = Nk / (Nk.sum() + eps)
            self.alive = torch.ones_like(pi, dtype=torch.bool)
            return pi

        prior = self.alpha - 1.0
        pi_unnorm = Nk + prior
        pi_unnorm = torch.clamp(pi_unnorm, min=1e-12)
        pi = pi_unnorm / (pi_unnorm.sum() + eps)

        if self.prune_eps > 0:
            dead = pi < self.prune_eps
            if dead.all():
                dead[torch.argmax(pi)] = False
            pi = pi.masked_fill(dead, 0.0)
            pi = pi / (pi.sum() + eps)
            self.alive = ~dead
        else:
            self.alive = torch.ones_like(pi, dtype=torch.bool)

        return pi

    # ---------------- core math ----------------

    @torch.no_grad()
    def _log_gauss_chunk(self, x: torch.Tensor, mu: torch.Tensor, W: torch.Tensor, sigma2: torch.Tensor):
        """
        Compute log N(x | mu_k, W_k W_k^T + sigma2_k I) for all k, chunked over x.

        x: (B,D) float32
        mu: (K,D) float32
        W: (K,D,d) float32
        sigma2: (K,) float32

        returns log_norm: (B,K) float32
        """
        B, D = x.shape
        K = mu.shape[0]
        d = W.shape[2]

        # precompute per-sample ||x - mu_k||^2 isn't cheap for all k; we do per-k loop
        # to avoid allocating (B,K,D).
        log_norm = torch.empty((B, K), device=x.device, dtype=torch.float32)

        const = D * math.log(2.0 * math.pi)

        for k in range(K):
            if self.alive is not None and (not bool(self.alive[k].item())):
                log_norm[:, k] = -1e9
                continue

            mu_k = mu[k]                 # (D,)
            W_k = W[k]                   # (D,d)
            s2 = sigma2[k].clamp_min(self.sigma2_min)

            xc = x - mu_k.unsqueeze(0)   # (B,D)
            # t = xc^T W_k  -> (B,d)
            t = xc @ W_k                 # (B,d)

            # A = I + (1/s2) W^T W   (d,d)
            # compute WtW
            WtW = W_k.t() @ W_k          # (d,d)
            A = torch.eye(d, device=x.device, dtype=torch.float32) + (WtW / s2)

            # solve A^{-1} t^T  -> (d,B) or do solve on (B,d) with transpose trick
            # We want per-sample: t A^{-1} t^T = sum(t * (t @ A^{-1}), dim=1)
            # Use: y = solve(A, t^T).T  => (B,d)
            y = torch.linalg.solve(A, t.t()).t()  # (B,d)

            # quad form using Woodbury:
            # q = (1/s2)||xc||^2 - (1/s2^2) * (t A^{-1} t^T)
            xc2 = torch.sum(xc * xc, dim=1)               # (B,)
            tAt = torch.sum(t * y, dim=1)                 # (B,)
            q = (xc2 / s2) - (tAt / (s2 * s2))

            # logdet:
            # log|C| = D log s2 + log|A|
            logdetA = torch.logdet(A.clamp_min(self.eig_eps))
            logdetC = D * torch.log(s2) + logdetA

            log_norm[:, k] = -0.5 * (const + logdetC + q)

        return log_norm

    # ---------------- API ----------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """
        X: (N,D) cuda
        """
        assert X.is_cuda and X.dim() == 2
        X = X.to(self.dtype)
        N, D = X.shape
        K = self.K
        d = self.d_latent
        eps = self.eps
        chunk = self.x_chunk

        # init mu via kmeans++
        mu = self._init_kmeanspp(X)  # (K,D)

        # init sigma2: global average variance
        var0 = X.var(dim=0, unbiased=False).mean().clamp_min(self.sigma2_min)  # scalar
        log_sigma2 = torch.full((K,), float(var0.log().item()), device=X.device, dtype=X.dtype)

        # init W: small random (or zeros)
        g = torch.Generator(device=X.device)
        g.manual_seed(self.seed + 123)
        W = 0.01 * torch.randn((K, D, d), generator=g, device=X.device, dtype=X.dtype)

        log_pi = torch.full((K,), math.log(1.0 / K), device=X.device, dtype=X.dtype)
        self.alive = torch.ones((K,), device=X.device, dtype=torch.bool)

        # choose accum dtype
        acc_dtype = torch.float32 if self.use_float32_accum else torch.float64

        for it in range(self.iters):
            # stats
            Nk = torch.zeros((K,), device=X.device, dtype=acc_dtype)
            S1 = torch.zeros((K, D), device=X.device, dtype=acc_dtype)
            S2 = torch.zeros((K, D, D), device=X.device, dtype=acc_dtype)

            # snapshot params in float32 for stable math
            mu_f = mu.to(torch.float32)
            W_f = W.to(torch.float32)
            sigma2_f = torch.exp(log_sigma2.to(torch.float32)).clamp_min(self.sigma2_min)  # (K,)
            log_pi_f = log_pi.to(torch.float32)

            # E-step (chunked)
            for s in range(0, N, chunk):
                x = X[s:s+chunk].to(torch.float32)  # (B,D)
                # compute log N(x|k)
                log_norm = self._log_gauss_chunk(x, mu_f, W_f, sigma2_f)  # (B,K)

                # mix
                log_pi_eff = log_pi_f.clone()
                if self.alive is not None:
                    log_pi_eff[~self.alive] = -1e9

                log_r = log_pi_eff.unsqueeze(0) + log_norm
                log_r = log_r - torch.logsumexp(log_r, dim=1, keepdim=True)
                r = torch.exp(log_r)  # (B,K)
                r = self._apply_topm(r)

                # accumulate stats
                r_acc = r.to(acc_dtype)
                x_acc = x.to(acc_dtype)

                Nk += r_acc.sum(dim=0)
                S1 += r_acc.t() @ x_acc
                # S2: sum_k r_ik * x_i x_i^T
                # compute x^T (r[:,k]*x) per k
                # do K-loop: K is small (32~128). This avoids (B,K,D) memory.
                for k in range(K):
                    if self.alive is not None and (not bool(self.alive[k].item())):
                        continue
                    w = r_acc[:, k].unsqueeze(1)       # (B,1)
                    xw = x_acc * w                     # (B,D)
                    S2[k] += x_acc.t() @ xw            # (D,D)

                del x, log_norm, log_r, r, r_acc, x_acc

            Nk = Nk.clamp_min(eps)

            # update pi (capacity control / pruning)
            pi = self._update_pi_with_capacity(Nk.to(torch.float32)).to(acc_dtype)
            log_pi = torch.log(pi.clamp_min(eps)).to(X.dtype)
            if self.alive is not None:
                log_pi[~self.alive] = -1e9

            # update mu
            mu_new = (S1 / Nk.unsqueeze(1)).to(torch.float32)  # (K,D)

            # update W and sigma2 per component from covariance eig
            W_new = torch.zeros((K, D, d), device=X.device, dtype=torch.float32)
            sigma2_new = torch.zeros((K,), device=X.device, dtype=torch.float32)

            for k in range(K):
                if self.alive is not None and (not bool(self.alive[k].item())):
                    # keep something valid
                    sigma2_new[k] = sigma2_f[k]
                    W_new[k] = W_f[k]
                    continue

                mk = mu_new[k].to(acc_dtype)  # (D,)
                # Cov = E[xx^T] - mu mu^T
                Exx = (S2[k] / Nk[k]).to(acc_dtype)  # (D,D)
                cov = Exx - mk.unsqueeze(1) @ mk.unsqueeze(0)
                cov = 0.5 * (cov + cov.t())  # enforce symmetry
                cov = cov.to(torch.float32)

                # eigen-decomp
                # eigh returns ascending eigenvalues
                evals, evecs = torch.linalg.eigh(cov)
                evals = torch.clamp(evals, min=self.eig_eps)

                # pick top-d (largest)
                if d >= D:
                    # degenerate: treat as full rank
                    d_eff = D - 1
                else:
                    d_eff = d

                top_evals = evals[-d_eff:]          # (d_eff,)
                top_evecs = evecs[:, -d_eff:]       # (D,d_eff)

                # sigma2 = mean of remaining eigenvalues
                if D - d_eff > 0:
                    sigma2_k = evals[:-d_eff].mean()
                else:
                    sigma2_k = evals.mean()

                sigma2_k = torch.clamp(sigma2_k, min=self.sigma2_min)

                # W = U (Lambda - sigma2 I)^{1/2}
                # clamp to avoid negative due to numeric
                lam_minus = torch.clamp(top_evals - sigma2_k, min=0.0)
                Wk = top_evecs * torch.sqrt(lam_minus).unsqueeze(0)  # (D,d_eff)

                # write
                W_new[k, :, :d_eff] = Wk
                sigma2_new[k] = sigma2_k

            # commit
            mu = mu_new.to(X.dtype)
            W = W_new.to(X.dtype)
            log_sigma2 = torch.log(sigma2_new.clamp_min(self.sigma2_min)).to(X.dtype)

            # optional: periodic prune stability
            if self.prune_eps > 0 and self.prune_every > 0 and (it + 1) % self.prune_every == 0:
                pi_now = torch.exp(log_pi.to(torch.float32)).clamp_min(0.0)
                pi_now = pi_now / (pi_now.sum() + eps)
                dead = pi_now < self.prune_eps
                if dead.all():
                    dead[torch.argmax(pi_now)] = False
                self.alive = ~dead
                pi_now = pi_now.masked_fill(dead, 0.0)
                pi_now = pi_now / (pi_now.sum() + eps)
                log_pi = torch.log(pi_now.clamp_min(eps)).to(X.dtype)
                log_pi[dead] = -1e9

        self.log_pi = log_pi
        self.mu = mu
        self.W = W
        self.log_sigma2 = log_sigma2
        self.fitted = True

    @torch.no_grad()
    def log_prob(self, X: torch.Tensor, mode = "all") -> torch.Tensor:
        """
        Return log p(x) chunked.

        mode:
        - "all" : full mixture logsumexp over all components
        - "topk": logsumexp over top-k components only

        X: (N,D) cuda
        """
        assert self.fitted and X.is_cuda and X.dim() == 2
        


        X = X.to(self.mu.dtype)
        chunk = self.x_chunk

        mu = self.mu.to(torch.float32)
        W = self.W.to(torch.float32)
        sigma2 = torch.exp(self.log_sigma2.to(torch.float32)).clamp_min(self.sigma2_min)
        log_pi = self.log_pi.to(torch.float32)

        out = []
        for s in range(0, X.shape[0], chunk):
            x = X[s:s+chunk].to(torch.float32)
            log_norm = self._log_gauss_chunk(x, mu, W, sigma2)  # (B,K)

            log_pi_eff = log_pi.clone()
            if self.alive is not None:
                log_pi_eff[~self.alive] = -1e9

            log_comp = log_pi_eff.unsqueeze(0) + log_norm   # (B,K)

            if mode=="all":
                lp = torch.logsumexp(log_comp, dim=1)

            elif mode == "topk":
                topk = self.topm
                K = log_comp.shape[1]
                assert topk is not None and topk > 0, "topk must be a positive int when mode='topk'"
                m = min(int(topk), K)

                if m >= K:
                    lp = torch.logsumexp(log_comp, dim=1)
                else:
                    vals, _ = torch.topk(log_comp, k=m, dim=1, largest=True, sorted=False)  # (B,m)
                    lp = torch.logsumexp(vals, dim=1)

            out.append(lp)
            del x, log_norm, log_comp, lp

        return torch.cat(out, dim=0)
    

    @torch.no_grad()
    def responsibilities_and_ell(self, X: torch.Tensor):
        """
        For a chunk batch X (2D cuda):
          r: (B,K)
          ell: (B,K)  where ell = -log N(x|k)  (component NLL, without mixing weights)
        """
        assert self.fitted and X.is_cuda and X.dim() == 2
        X = X.to(torch.float32)

        mu = self.mu.to(torch.float32)
        W = self.W.to(torch.float32)
        sigma2 = torch.exp(self.log_sigma2.to(torch.float32)).clamp_min(self.sigma2_min)
        log_pi = self.log_pi.to(torch.float32)

        log_norm = self._log_gauss_chunk(X, mu, W, sigma2)  # (B,K)

        log_pi_eff = log_pi.clone()
        if self.alive is not None:
            log_pi_eff[~self.alive] = -1e9

        log_r = log_pi_eff.unsqueeze(0) + log_norm
        log_r = log_r - torch.logsumexp(log_r, dim=1, keepdim=True)
        r = torch.exp(log_r)
        r = self._apply_topm(r)

        ell = -log_norm
        return r, ell



class GMM_AD_Wrapper:
    """
      - collect() gathers tokens (cpu)
      - finalize() fits GMM_AD on cuda
      - score() returns token NLL (-log p(x))
    """
    def __init__(
        self,
        K=32,
        d_latent=8,
        iters=30,
        max_tokens=1_000_000,
        seed=0,
        device="cuda",
        x_chunk=65536,
        dtype=torch.float32,
        topm=None,
        alpha=1.0,
        prune_eps=0.0,
        prune_every=5,
        sigma2_min=1e-4,
    ):
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.device = torch.device(device)

        self._buf = []
        self.gmm_ad = GMM_AD(
            K=K, d_latent=d_latent, iters=iters, seed=seed,
            x_chunk=x_chunk, dtype=dtype,
            topm=topm, alpha=alpha, prune_eps=prune_eps, prune_every=prune_every,
            sigma2_min=sigma2_min
        )
        self.ready = False

    @torch.no_grad()
    def collect(self, y: torch.Tensor):
        # y: (B,N,D) or (N,D)
        if y.dim() == 3:
            y = y.reshape(-1, y.shape[-1])
        self._buf.append(y.detach().float().cpu())

    def finalize(self):
        import numpy as np
        assert len(self._buf) > 0, "collect() some tokens first"
        X = torch.cat(self._buf, dim=0).numpy()
        self._buf.clear()

        M = X.shape[0]
        if M > self.max_tokens:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(M, size=self.max_tokens, replace=False)
            X = X[idx]

        Xt = torch.from_numpy(X.astype("float32")).to(self.device)
        self.gmm_ad.fit(Xt)
        self.ready = True

    @torch.no_grad()
    def score(self, y: torch.Tensor) -> torch.Tensor:
        assert self.ready
        dev = y.device
        if y.dim() == 3:
            B, N, D = y.shape
            Y = y.reshape(-1, D)
        else:
            B = None
            N, D = y.shape
            Y = y

        Xt = Y.detach().float().to(self.device)
        logp = self.gmm_ad.log_prob(Xt)   # (BN,)
        nll = (-logp).to(dev)

        if B is None:
            return nll.view(N)
        return nll.view(B, N)


