import numpy as np
import numpy.ctypeslib as ctl
import matplotlib.colors as colors
import matplotlib.pyplot as plt


#import scipy.io as sio
#from scipy.stats import poisson, norm, beta

from fastpsf import Context#, Estimator
from fastpsf import GaussianPSFMethods, Gauss3D_Calibration
from fastpsf.mcmc import MCLocalizer, PDFInfo

from priors_release import Prior_calculator, gauss1d, prior_k, prior_I, prior_bg, prior_xyz
from moves_release import RJMCMC_Object

np.seterr(divide='ignore', invalid='ignore')

#%% create PDF

k_max = 6     # maxEmitters
k_PDF = np.ones(k_max)
k_PDF /= k_PDF.sum()

bg_PDF = np.zeros(1000)
bg_PDF[0:30] = 1
bg_PDF /= bg_PDF.sum()


# synthetic PDFs for I, gaussians
s_I = 300
b_I = int(3*s_I) 
I_PDF = gauss1d(np.arange(30000), 2000, s_I)
I_PDF[2000 + b_I::] = 0
frac = 40
I_PDF[b_I:np.where(I_PDF > max(I_PDF)/frac)[0][0]] = max(I_PDF)/frac
I_PDF[0:b_I] = np.linspace(0,max(I_PDF)/frac, b_I)
I_PDF /= I_PDF.sum()

np.random.seed(0)       #debug

with Context() as ctx:
    
    # generate astigmatic psf model
    sigma_PSF = 1.2
    fixed_I = 2000
    bg = 20
    gpm = GaussianPSFMethods(ctx)
    
    #sigma_PSF = 1.2 #bamf
    x=[sigma_PSF,  2, 3, 0]
    y=[sigma_PSF, -2, 3, 0]
    zrange=[-1.3, 1.3]
    roisize = 20  

    ################# experimental setup params ###################
    #zrange=[-1.0, 1.0]
    #sigma_PSF = 1.95
    #x = [1.985221028327942, -0.2022230178117752, 0.2916024327278137, 0.07859058678150177]
    #y = [1.935728669166565, 0.22407332062721252, 0.3378242552280426, -0.42143693566322327]
    #roisize = 30
    ###############################################################

    calib = Gauss3D_Calibration(x,y,zrange)     # [s0_x, gamma_x, d_x, A_x], [s0_y, gamma_y, d_y, A_y])
    psf = gpm.CreatePSF_XYZIBg(roisize, calib, cuda=True) 
   
    # parameters for the data
    n_eval = 50              
    numFrames = 1
    numEmitters = 2
    
    
    
    j = 5
    separation = sigma_PSF*j/8 # dist to COM. dist between emitters is 2*separation

    bg = 20 
    border = 4
    
    # set up likelihood evaluator 
    intensityPDF = PDFInfo(I_PDF[100:3000],100, 3000)    
    mcl = MCLocalizer(k_max, numFrames, psf, intensityPDF, ctx) 

    # generate parameters, images
    pos = np.zeros([n_eval, k_max, 3], dtype = np.float32)
    pos[:,0,0] = roisize / 2 - separation
    pos[:,1,0] = roisize / 2 + separation
    pos[:,0:2,1] = roisize / 2

    I = np.zeros([n_eval, k_max, numFrames], dtype = np.float32)
    I[:,0:2,:] = 2000

    emitterState = np.zeros([n_eval, k_max], dtype = bool)
    emitterState[:,0:2] = True
    backgroundParams = np.ones(n_eval)*bg
    
    expval = mcl.ExpectedValue(pos, I*emitterState[:,:,None]) + bg
    smp = np.random.poisson(expval)
    smpIndices = np.arange(n_eval, dtype = int)

    # assign samples to ll calculator
    mcl.SetSamples(smp, smp*0)     

    fig, ax = plt.subplots(dpi=300)
    plt.imshow(smp[0,0,:,:])
    plt.scatter(pos[0, emitterState[0,:], 0], pos[0, emitterState[0,:], 1], marker='.', color='b', alpha = .5)
    plt.legend(['GT'])
    plt.title('Sample frame')

    #%% set up RJMCMC localization

    RJ = RJMCMC_Object(ctx, psf, smp, zrange, sigma_PSF, z_rescale = 1)
    RJ.set_priors(k_max = k_max)
    
    # I_PDF, k_PDF
    RJ.I_PDF = I_PDF
    RJ.k_PDF = k_PDF
    RJ.p_calc = Prior_calculator(RJ.n_eval, prior_I, RJ.I_PDF, prior_bg, RJ.bg_PDF,
                                   prior_xyz, RJ.roisize, RJ.border, RJ.z_range[0], 
                                   RJ.z_range[1], prior_k, RJ.k_PDF, RJ.mu_k, 
                                   RJ.k_max)
    RJ.intensityPDF = PDFInfo(RJ.I_PDF[100:3000],100, 3000) 
    
    RJ.set_rjmcmc_moves(rjmcmc_len = 30000, rjmcmc_burn = 20000, sigma_z = 0.08, sigma_I = 15)
    RJ.set_mcmc_moves(mcmc_len = 5000, sigma_z = 0.07,  sigma_I = 15)
    
    #%% run RJMCMC with default settings (5k rj, 3k mc, bamf move probabilities)
    
    RJ.run_rjmcmc()
    
    #%% run MCMC with default settings
    
    RJ.select_MAPN()
    RJ.run_mcmc()

#%% reconstruction

fig, ax = plt.subplots(dpi=300)
#plt.title(f'MCMC reconstruction using MAPN model, {n_eval} frames')
b0, b1, b2 = np.where(RJ.mcmc.state_chain)
x_mcmc = RJ.mcmc.pos_chain[b0, b1, 0, b2]
y_mcmc = RJ.mcmc.pos_chain[b0, b1, 1, b2]
z_mcmc = RJ.mcmc.pos_chain[b0, b1, 2, b2]
plt.hist2d(x_mcmc, y_mcmc, bins = np.linspace(0,roisize, 20*roisize + 1), cmap = 'hot')
plt.title(f'Histogram reconstruction, aggregate of {n_eval} frames')
    
fig, ax = plt.subplots(dpi=300)
plt.hist2d(x_mcmc, y_mcmc, bins = np.linspace(roisize/2-4,roisize/2+4, 20*8 + 1), cmap = 'hot')
plt.scatter(pos[0, emitterState[0,:], 0], pos[0, emitterState[0,:], 1], marker='.', color='b', alpha = .5)
plt.legend(['GT'])
plt.title('Zoomed in reconstruction with ground truth')


################# XZ  ########################

fig, ax = plt.subplots(dpi=300)
plt.hist2d(x_mcmc, z_mcmc, bins = [np.linspace(0,roisize, 20*roisize + 1), np.linspace(zrange[0], zrange[1], 20*roisize+1)], cmap = 'hot')
plt.title('XZ plane reconstruction')
################## ZY #######################

fig, ax = plt.subplots(dpi=300)
plt.hist2d(z_mcmc, y_mcmc, bins = [np.linspace(zrange[0], zrange[1], 20*roisize+1), np.linspace(0,roisize, 20*roisize + 1)], cmap = 'hot')
plt.title('ZY plane reconstruction')


minimum_distance_xy = .25 
minimum_count = 100

average_model_accuracy = (RJ.found_emitters == emitterState.sum(-1)).sum()/n_eval  

fig, ax = plt.subplots(dpi=300)
plt.plot(np.concatenate([RJ.rjmcmc.state_chain_burn, RJ.rjmcmc.state_chain],-1).sum(1).T)
plt.xlabel('iteration')
plt.ylabel('number of emitters')
plt.title('model time series for each frame')


indices = np.arange(n_eval)
cm = 'viridis'
cm = 'hot'


fig, ax = plt.subplots(dpi=300)
plt.plot(RJ.I_PDF[0:2000+b_I])
plt.xlabel('I [photons]')
plt.ylabel('P(I)')
plt.grid()
plt.title('Emitter intensity prior')
