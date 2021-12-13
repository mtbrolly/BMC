def evidenceBpost(tau, dt, Xdata, kappa_mean, Np, Ntau):
    import numpy as np
    from Brownianf import (logPostB, max_lPostB, dkk_logLikeB, log_exp_prior,
                           logLikeB)

    obsInt = int(tau / dt)
    Xobs = Xdata[:, :1 + Ntau * obsInt:obsInt, :Np]
    DX = (Xobs[:, 1:, :] - Xobs[:, :-1, :])

    def nlPost(kappa):
        return -logPostB(DX, kappa, tau)

    def dkk_nlLike(kappa):
        return -dkk_logLikeB(DX, kappa, tau)

    kappa_post_star, maxlPB = max_lPostB(Xobs, tau, kappa_mean)
    maxlLB = logLikeB(DX, kappa_post_star, tau)

    log_prior = log_exp_prior(kappa_post_star, kappa_mean)
    obsInfo = dkk_nlLike(kappa_post_star)
    postVarKappa = obsInfo ** (-1)
    lnOccamB = log_prior - 0.5 * np.log(obsInfo / (2 * np.pi))
    lnLapEviB = maxlLB + lnOccamB

    return maxlLB, kappa_post_star, lnLapEviB, postVarKappa
