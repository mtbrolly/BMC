def evidenceLmargpost(tau, dt, Xdata, g_mean, k_mean, Np, Ntau, h=1e-2):
    import numpy as np
    from inference.Langevinf import (max_lPostLmarg, hess_logPostLmarg,
                                     log_exp_prior, logLikeLmarg)

    obsInt = int(tau / dt)
    Xobs = Xdata[:, :1 + Ntau*obsInt:obsInt, :Np]

    def hess_nlPost(theta):
        return -hess_logPostLmarg(Xobs, theta[0], theta[1], tau,
                                  g_mean, k_mean)

    theta_star, maxlPL = max_lPostLmarg(Xobs, tau, g_mean, k_mean, h=h)
    maxlLL = logLikeLmarg(Xobs, theta_star[0], theta_star[1], tau)
    obsInfo = hess_nlPost(theta_star)
    postVar = np.linalg.inv(obsInfo)

    log_prior = log_exp_prior(theta_star[0], theta_star[1], g_mean, k_mean)
    lnOccamL = log_prior - 0.5 * np.log(np.linalg.det(obsInfo
                                                      / (2 * np.pi)))
    lnLapEviL = maxlLL + lnOccamL
    lnEviL = lnLapEviL

    return maxlLL, theta_star, lnEviL, postVar
