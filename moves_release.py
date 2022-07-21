import numpy as np
import ctypes as ct
import numpy.ctypeslib as ctl
from scipy.stats import poisson, beta

from fastpsf import Context, Estimator
from priors_release import gauss1d, Prior_calculator, prior_I, prior_xyz, prior_bg, prior_k

from fastpsf.mcmc import MCLocalizer, PDFInfo

def closestMulti2(pos, emitterState, loc, emIndices, dmin, pixelSize = 100, z_rescale = 1):
    
    n_eval = emitterState.shape[0]
    maxEmitters = emitterState.shape[1]
    emstate_temp = np.copy(emitterState)
    emstate_temp[loc, emIndices] = False
    mp_temp = emstate_temp.sum(1)
    x0 = pos[loc, emIndices, 0]
    y0 = pos[loc, emIndices, 1]
    z0 = pos[loc, emIndices, 2]
    
    x2 = np.zeros(emitterState.shape)
    y2 = np.zeros(emitterState.shape)
    z2 = np.zeros(emitterState.shape)
    x2[:,:] = np.nan
    y2[:,:] = np.nan
    z2[:,:] = np.nan

    for i in range(len(loc)):
        if mp_temp[i] != 0:
            x2[i,emstate_temp[i]] = np.square(pos[i, emstate_temp[i],0] - x0[i])
            y2[i,emstate_temp[i]] = np.square(pos[i, emstate_temp[i],1] - y0[i])
            z2[i,emstate_temp[i]] = np.square((pos[i, emstate_temp[i],2] - z0[i])*1000/pixelSize*z_rescale)

    d = np.sqrt(x2 + y2 + z2)  
    
        
    Ind = np.zeros(emitterState.shape, dtype = bool)
    Ind[np.where(d < dmin)] = True
    Num = Ind.sum(1)

    Np = np.zeros(n_eval, dtype = int)
    Ua = np.random.uniform(0,1,(Num > 0).sum()) 
    Np2 = Np[Num > 0]
    Num2 = Num[Num > 0]
    rand_ind = Ua < 1/Num2
    Np2[rand_ind] = Num2[rand_ind]
    Np2[~rand_ind] = np.amin(((Ua*Num2)[~rand_ind].astype(int), Num2[~rand_ind] - 1), axis = 0)
    
    Np[Num > 0] = Np2
    
    # for sets with Num > 0
    # for 0 < i < Np
    # randomly pick Np elements from Num closest elements to return as cluster
    Closest = np.zeros(emitterState.shape, dtype = bool)

    for i in range(len(loc)):
        Closest[i, np.random.choice(np.where(Ind[i])[0], Np[i], replace = False)] = True
        Closest[i, emIndices[i]] = True
    return Closest

# how should intensities be jumped for high number of frames? becomes 
# harder and harder to push intensities over multiple frames in the right 
# direction simultaneously

class Single_Emitter_Move:
    def __init__(self, sigma_x, sigma_y, sigma_z, sigma_I, prior_func, LL_func):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_I = sigma_I
        self.sigma_z = sigma_z
        self.prior_func = prior_func
        self.LL_func = LL_func
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        n_eval, km, numFrames = I.shape
        # in each frame, pick an emitter to update
        update_indices = np.zeros(n_eval, dtype = int)
        for i in range(n_eval):
            if emitterState[i].sum() != 0:
                update_indices[i] = np.random.choice(np.arange(k_max)[emitterState[i]])     
            else:
                update_indices[i] = 0
                
        # move a single element (x,y,z,I) for each sample
        pos_update = np.copy(pos)
        I_update = np.copy(I)
        pos_update[range(n_eval),update_indices,0] += np.random.normal(0, self.sigma_x, n_eval)
        pos_update[range(n_eval),update_indices,1] += np.random.normal(0, self.sigma_y, n_eval)
        pos_update[range(n_eval),update_indices,2] += np.random.normal(0, self.sigma_z, n_eval)
        
        # intensities over all frames randomly
        I_update[range(n_eval),update_indices, :] += np.random.normal(0, self.sigma_I, [n_eval, numFrames])
        
        # calculate posterior ratio
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState[:,:,None], backgroundParams, smpIndices)

        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)
        u = np.random.uniform(0,1,n_eval)

        
        failed_updates = (PR < u) + np.isnan(PR)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]        # rejected jumps in each dataset get set back to their original values
        I_update[failed_updates] = I[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState, ~failed_updates, ll_new, prior_new 

class Group_Move_Unconserved:
    def __init__(self, sigma_PSF, sigma_x, sigma_y, sigma_z, sigma_I, prior_func, LL_func, z_rescale = 1):
        self.sigma_PSF = sigma_PSF
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.sigma_I = sigma_I
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.z_rescale = z_rescale
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        n_eval, km, numFrames = I.shape
        modelParams = emitterState.sum(1)
        loc = np.where(modelParams > 1)[0]
        if len(loc) == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev

        # in each frame, pick an emitter to update
        update_indices = np.zeros(len(loc), dtype = int)
        for i in range(len(loc)):
            if emitterState[i].sum() != 0:
                update_indices[i] = np.random.choice(np.arange(k_max)[emitterState[i]])     
            else:
                update_indices[i] = 0

        Closest = closestMulti2(pos, emitterState, loc, update_indices, 2*self.sigma_PSF, z_rescale = self.z_rescale)

        a1 = np.where(Closest[loc])[0]
        a2 = np.where(Closest[loc])[1]
        moves = Closest[loc].sum()

        pos_update = np.copy(pos)
        I_update = np.copy(I)
        pos_update[a1,a2,0] += np.random.normal(0, self.sigma_x, moves)
        pos_update[a1,a2,1] += np.random.normal(0, self.sigma_y, moves)
        pos_update[a1,a2,2] += np.random.normal(0, self.sigma_z, moves)
        
        # moving I over all frames randomly
        I_update[a1,a2,:] += np.random.normal(0, self.sigma_I, [moves, numFrames])

        # calculate posterior ratio
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState[:,:,None], backgroundParams, smpIndices)

        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)
        u = np.random.uniform(0,1,n_eval)

        
        failed_updates = (PR < u) + np.isnan(PR)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]        # rejected jumps in each dataset get set back to their original values
        I_update[failed_updates] = I[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState, ~failed_updates, ll_new, prior_new  
         
        
class Background_Move:
    def __init__(self, sigma_bg, sigma_ax, sigma_ay, prior_func, LL_func):
        self.sigma_bg = sigma_bg
        self.sigma_ax = sigma_ax
        self.sigma_ay = sigma_ay
        self.prior_func = prior_func
        self.LL_func = LL_func
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        n_eval = emitterState.shape[0]
        
        # jump background parameters
        backgroundParams_update = np.copy(backgroundParams) + np.random.normal(0, self.sigma_bg, n_eval)
        
        # calculate posterior ratio
        prior_new = self.prior_func.calc(pos, I, backgroundParams_update, emitterState, smpIndices)
        ll_new = self.LL_func.Likelihood(pos, I*emitterState[:,:,None], backgroundParams_update, smpIndices)

        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)
        u = np.random.uniform(0,1,n_eval)
        
        failed_updates = (PR < u) + np.isnan(PR)        # where posterior ratio < random number AND where posterior ratio isNaN
        backgroundParams_update[failed_updates] = backgroundParams[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos, I, backgroundParams_update, emitterState, ~failed_updates, ll_new, prior_new
     


class Split:
    def __init__(self, sigma_PSF, prior_func, LL_func, p_split, p_merge, z_rescale = 1):
        self.sigma_PSF = sigma_PSF
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.p_merge = p_merge
        self.p_split = p_split
        self.z_rescale = z_rescale
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        modelParams = emitterState.sum(1)
        n_eval, km, numFrames = I.shape
        
        # evaluate where modelParams < k_max
        splits = len(modelParams[modelParams < k_max])
        if splits == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev #, 'split failed: all sets are at k_max'
        
        loc = np.where(modelParams < k_max)[0]         # splitable sets
        # random numbers for split:
        u1 = np.random.beta(1,1, splits)
        u2 = np.random.normal(0, self.sigma_PSF, splits)         
        u3 = np.random.normal(0, self.sigma_PSF, splits)
        
        u4 = np.random.normal(0, self.sigma_PSF*self.z_rescale, splits)        # additional random term for axial position
        
        
        # pick an emitter to split
        j1 = np.zeros(len(loc), dtype = int)
        j2 = np.zeros(len(loc), dtype = int)
        emitterState_update = np.copy(emitterState)
        for i in range(len(loc)):
            if emitterState[loc[i]].sum() != 0:
                j1[i] = np.random.choice(np.arange(0, k_max)[emitterState[loc[i]]])     # emitter to split
                j2[i] = np.arange(0, k_max)[~emitterState[loc[i]]][0]                   # empty emitterState to move the newly generated emitter into
                emitterState_update[loc[i],j2[i]] = True
            else:
                j1[i] = 0
                j2[i] = 1
                
        # split a single element in each sample
        pos_update = np.copy(pos)
        I_update = np.copy(I)
        
        # I_j1 = Ij * u1
        # I_j2 = Ij * (1 - u1)
        # multi-frame: each frame separately splits its intensity as normal
        I_update[loc, j1, :] = I[loc, j1, :] * u1[:,None]
        I_update[loc, j2, :] = I[loc, j1, :] * (1 - u1)[:,None]
        
        # x_j1 = x_j + u2
        # x_j2 = x_j - (u1 * u2) / (1 - u1)
        pos_update[loc,j1,0] += u2
        pos_update[loc,j2,0] = pos[loc,j1,0] - (u1 * u2) / (1 - u1)
        
        # y_j1 = y_j + u3
        # y_j2 = y_j - (u1 * u3) / (1 - u1)
        pos_update[loc,j1,1] += u3
        pos_update[loc,j2,1] = pos[loc,j1,1] - (u1 * u3) / (1 - u1)
        
        # 3d: z-positions conserving COM
        pos_update[loc,j1,2] += u4
        pos_update[loc,j2,2] = pos[loc,j1,2] - (u1 * u4) / (1 - u1)

        # calculate the acceptance rate
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState_update, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState_update[:,:,None], backgroundParams, smpIndices)

        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)/(self.prior_func.roisize + 2*self.prior_func.border)**2*(self.prior_func.mu_k/emitterState_update.sum(1))
        u = np.random.uniform(0,1,n_eval)
        
        # additional term from probability of drawing u4 from its distribution
        q = np.zeros(n_eval)
        q[loc] = gauss1d(u2, 0, self.sigma_PSF) * gauss1d(u3, 0, self.sigma_PSF) * gauss1d(u4, 0, self.sigma_PSF*self.z_rescale) * beta.pdf(u1,1,1)


        det_jac = np.zeros([n_eval, numFrames])
        det_jac[loc, :] = I[loc, j1, :] / ((1 - u1)**3)[:,None]
        
        A = np.zeros(n_eval)
        A[loc] = PR[loc] * self.p_merge/self.p_split / q[loc] * det_jac.prod(1)[loc]        

        failed_updates = (A < u) + np.isnan(A)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]
        I_update[failed_updates] = I[failed_updates]
        emitterState_update[failed_updates] = emitterState[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState_update, ~failed_updates, ll_new, prior_new
        
class Generalized_Split:
    def __init__(self, sigma_PSF, prior_func, LL_func, p_split, p_merge, z_rescale = 1):
        self.sigma_PSF = sigma_PSF
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.p_merge = p_merge
        self.p_split = p_split
        self.z_rescale = z_rescale
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        modelParams = emitterState.sum(1)
        n_eval, km, numFrames = I.shape
        
        delete = np.array([], dtype = int)
        loc = np.where(modelParams < k_max)[0]          # sets with less than maxEmitters
        for i in np.arange(len(loc)):
            if modelParams[loc[i]] < 2:
                delete = np.append(delete, i)           # sets with less than 2 active emitters are not viable for g-split
        loc = np.delete(loc, delete)  
        
        if len(loc) == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev 

        # pick an emitter to split
        j1 = np.zeros(len(loc), dtype = int)
        j2 = np.zeros(len(loc), dtype = int)
        emitterState_update = np.copy(emitterState)

        for i in range(len(loc)):
            if emitterState[loc[i]].sum() != 0:
                j1[i] = np.random.choice(np.arange(0, k_max)[emitterState[loc[i]]])
                j2[i] = np.arange(0, k_max)[~emitterState[loc[i]]][0]
                emitterState_update[loc[i],j2[i]] = True
            else:
                j1[i] = 0
                j2[i] = 1

        Closest = closestMulti2(pos, emitterState, loc, j1, 2*self.sigma_PSF, z_rescale = self.z_rescale)
        Np = Closest.sum(1)

        delete = np.array([], dtype = int)
        for i in np.arange(len(loc)):
            if Np[i] <= 1:                              # no emitters close to j1, only one element in the cluster
                delete = np.append(delete, i)
                
        emitterState_update[loc[delete], j2[delete]] = False
        loc = np.delete(loc, delete)

        j1 = np.delete(j1, delete)
        j2 = np.delete(j2, delete)

        # random numbers for split:
        u1 = np.random.beta(1,1, len(loc))
        u2 = np.random.normal(0, self.sigma_PSF, len(loc))         
        u3 = np.random.normal(0, self.sigma_PSF, len(loc))
        u4 = np.random.normal(0, self.sigma_PSF*self.z_rescale, len(loc))

        # split a single element from the groups in each sample:
        pos_update = np.copy(pos)
        pos_temp = np.copy(pos)
        I_update = np.copy(I)
        I_temp = np.copy(I)
        
        # rescale emitters in group with (1 - u1)
        I_temp[loc,:,:] = I[loc,:,:] * Closest[loc,:,None] * (1 - u1[:,None, None])

        # generate new emitter with intensity u1 * sum(I_closest)
        I_tot = (I[loc,:,:]*Closest[loc,:,None]).sum(1)
        I_update[loc,j2,:] = I_tot * u1[:,None]


        # x_i' = (x_i - u1(1/N*sum_1^N (x_i + u2))) / (1 - u1)        
        pos_temp[loc,:,0] = (pos[loc,:,0]*Closest[loc] - (((pos[loc,:,0] + u2[:,None])*Closest[loc]).sum(1)/Np[loc] * u1)[:,None]) / (1 - u1[:,None])
        pos_update[loc,j2,0] = ((pos[loc,:,0] + u2[:,None])*Closest[loc]).sum(1)/Np[loc]
        
        # y
        pos_temp[loc,:,1] = (pos[loc,:,1]*Closest[loc] - (((pos[loc,:,1] + u3[:,None])*Closest[loc]).sum(1)/Np[loc] * u1)[:,None]) / (1 - u1[:,None])
        pos_update[loc,j2,1] = ((pos[loc,:,1] + u3[:,None])*Closest[loc]).sum(1)/Np[loc]
        
        # z
        pos_temp[loc,:,2] = (pos[loc,:,2]*Closest[loc] - (((pos[loc,:,2] + u4[:,None])*Closest[loc]).sum(1)/Np[loc] * u1)[:,None]) / (1 - u1[:,None])
        pos_update[loc,j2,2] = ((pos[loc,:,2] + u4[:,None])*Closest[loc]).sum(1)/Np[loc]
        
        
        a1 = np.where(Closest[loc])[0]
        a2 = np.where(Closest[loc])[1]
        
        #emitterParams_update[a1, a2, -1] = emitterParams_temp[a1, a2, -1]
        I_update[a1,a2,:] = I_temp[a1,a2,:]
        pos_update[a1, a2, 0] = pos_temp[a1, a2, 0]
        pos_update[a1, a2, 1] = pos_temp[a1, a2, 1]
        pos_update[a1, a2, 2] = pos_temp[a1, a2, 2]
        
        
        
        # calculate the acceptance rate
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState_update, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState_update[:,:,None], backgroundParams, smpIndices)


        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)/(self.prior_func.roisize + 2*self.prior_func.border)**2*(self.prior_func.mu_k/emitterState_update.sum(1))
        u = np.random.uniform(0,1,n_eval)
        
        # extra term for 3d move accounting for u4 probability
        q = np.zeros(n_eval)
        q[loc] = gauss1d(u2, 0, self.sigma_PSF) * gauss1d(u3, 0, self.sigma_PSF) * gauss1d(u4, 0, self.sigma_PSF*self.z_rescale) * beta.pdf(u1,1,1)

        det_jac = np.zeros([n_eval, numFrames])
        det_jac[loc, :] = I_tot / ((1 - u1)**(Np[loc]*2 + 1))[:,None]
        
        A = np.zeros(n_eval)

        A[loc] = PR[loc] * self.p_merge/self.p_split / q[loc] * det_jac.prod(1)[loc]
        
        failed_updates = (A < u) + np.isnan(A)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]
        I_update[failed_updates] = I[failed_updates]
        emitterState_update[failed_updates] = emitterState[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState_update, ~failed_updates, ll_new, prior_new
    
class Merge:
    def __init__(self, sigma_PSF, prior_func, LL_func, p_merge, p_split, z_rescale = 1):
        self.sigma_PSF = sigma_PSF
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.p_merge = p_merge
        self.p_split = p_split
        self.z_rescale = z_rescale
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        modelParams = emitterState.sum(1)
        n_eval, km, numFrames = I.shape
        
        # evaluate where modelParams >= 2
        merges = len(modelParams[modelParams >= 2])
        if merges == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev
        loc = np.where(modelParams >= 2)[0]         # mergable sets
       
        # pick two emitters to merge
        j1 = np.zeros(len(loc), dtype = int)
        j2 = np.zeros(len(loc), dtype = int)
        emitterState_update = np.copy(emitterState)
        for i in range(len(loc)):
            if emitterState[loc[i]].sum() != 0:
                j2[i] = np.random.choice(np.arange(0, k_max)[emitterState[loc[i]]])     # pick elem to merge, index j2 turns off
                emitterState_update[loc[i],j2[i]] = False
                j1[i] = np.random.choice(np.arange(0, k_max)[emitterState_update[loc[i]]])     # pick other elem to merge
            else:
                j2[i] = 0
                j1[i] = 1
                
        # merge two elements in each sample
        pos_update = np.copy(pos)
        I_update = np.copy(I)
        
        # I_j = I_j1 + I_j2
        Ij1 = I[loc, j1, :]
        Ij2 = I[loc, j2, :]
        
        I_update[loc, j1, :] = Ij1 + Ij2
        I_update[loc, j2, :] = 0

        xj1 = pos[loc, j1, 0]
        xj2 = pos[loc, j2, 0]
        yj1 = pos[loc, j1, 1]
        yj2 = pos[loc, j2, 1]
        zj1 = pos[loc, j1, 2]
        zj2 = pos[loc, j2, 2]
        
        # x_j = (I_j1 * x_j1 + I_j2 * x_j2) / I_j
        # multi-frame: for CoM calculation, sum intensities of all frames for 
        # corresponding emitters and multiply with position, divide by total
        # intensity over all frames
        pos_update[loc, j1, 0] = (Ij1.sum(1) * xj1 + Ij2.sum(1) * xj2) / I_update[loc, j1, :].sum(1)
        pos_update[loc, j2, 0] = 0
        
        pos_update[loc, j1, 1] = (Ij1.sum(1) * yj1 + Ij2.sum(1) * yj2) / I_update[loc, j1, :].sum(1)
        pos_update[loc, j2, 1] = 0
        
        pos_update[loc, j1, 2] = (Ij1.sum(1) * zj1 + Ij2.sum(1) * zj2) / I_update[loc, j1, :].sum(1)
        pos_update[loc, j2, 2] = 0
        
        u1 = Ij1.sum(1) / I_update[loc, j1, :].sum(1)
        u2 = xj1 - pos_update[loc, j1, 0]
        u3 = yj1 - pos_update[loc, j1, 1]
        u4 = zj1 - pos_update[loc, j1, 2]
        
        
        # calculate the acceptance rate
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState_update, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState_update[:,:,None], backgroundParams, smpIndices)

        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)*(self.prior_func.roisize + 2*self.prior_func.border)**2/(self.prior_func.mu_k/emitterState_update.sum(1))
        u = np.random.uniform(0,1,n_eval)
                
        q = np.zeros(n_eval)
        q[loc] = gauss1d(u2, 0, self.sigma_PSF) * gauss1d(u3, 0, self.sigma_PSF) * gauss1d(u4, 0, self.sigma_PSF*self.z_rescale) * beta.pdf(u1,1,1)
        
        # I_j / (1 - u1)**2
        det_jac = np.zeros(n_eval)
        det_jac[loc] = (I_update[loc, j1, :] / ((1 - u1)**3)[:,None]).prod(1)       # multiplying probabilities over all frames
        
        A = np.zeros(n_eval)
        A[loc] = PR[loc] * self.p_split/self.p_merge * q[loc] / det_jac[loc]
                
        failed_updates = (A < u) + np.isnan(A)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]
        I_update[failed_updates] = I[failed_updates]
        emitterState_update[failed_updates] = emitterState[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState_update, ~failed_updates, ll_new, prior_new
       
 
class Generalized_Merge:
    def __init__(self, sigma_PSF, prior_func, LL_func, p_merge, p_split, z_rescale = 1):
        self.sigma_PSF = sigma_PSF
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.p_merge = p_merge
        self.p_split = p_split
        self.z_rescale = z_rescale
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        
        modelParams = emitterState.sum(1)
        n_eval = len(modelParams)

        loc = np.where(modelParams >= 3)[0]         # sets need at least 3 emitters for g-merge
        if len(loc) == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev 

        # pick an emitter to merge
        j2 = np.zeros(len(loc), dtype = int)
        emitterState_update = np.copy(emitterState)

        for i in range(len(loc)):
            if emitterState[loc[i]].sum() != 0:
                j2[i] = np.random.choice(np.arange(0, k_max)[emitterState[loc[i]]])     # merge j2, turns off
                emitterState_update[loc[i],j2[i]] = False
            else:
                j2[i] = 0
                
        Closest = closestMulti2(pos, emitterState, loc, j2, 3*self.sigma_PSF, z_rescale = self.z_rescale)    #  3*sigma or 2*sigma?

        Np = Closest.sum(1)
        
        delete = np.array([], dtype = int)
        for i in range(len(loc)):
            if Np[i] <= 1:                              # no emitters close to j2, only one element in the cluster
                delete = np.append(delete, i)

        
        
        emitterState_update[loc[delete], j2[delete]] = True
        loc = np.delete(loc, delete)
        j2 = np.delete(j2, delete)
        
        # merge elements in each sample
        pos_update = np.copy(pos)
        pos_temp = np.copy(pos)
        I_update = np.copy(I)
        I_temp = np.copy(I)
        

        # intensity
        u1 = I[loc,j2,:].sum(1) / (I[loc,:,:]*Closest[loc,:,None]).sum(1).sum(1)   # combined intensity of all em in cluster

        I_temp[loc, :,:] /= (1 - u1)[:,None,None]          
        Closest[loc,j2] = False

        # x
        pos_temp[loc,:,0] = u1[:,None] * pos[loc,j2,0][:,None] + (1 - u1)[:,None] * pos[loc,:,0]
        u2 = pos[loc,j2,0] - (1/(Np[loc] - 1))*(pos_temp[loc,:,0]*Closest[loc]).sum(1)

        # y
        pos_temp[loc,:,1] = u1[:,None] * pos[loc,j2,1][:,None] + (1 - u1)[:,None] * pos[loc,:,1]
        u3 = pos[loc,j2,1] - (1/(Np[loc] - 1))*(pos_temp[loc,:,1]*Closest[loc]).sum(1)

        # z
        pos_temp[loc,:,2] = u1[:,None] * pos[loc,j2,2][:,None] + (1 - u1)[:,None] * pos[loc,:,2]
        u4 = pos[loc,j2,2] - (1/(Np[loc] - 1))*(pos_temp[loc,:,2]*Closest[loc]).sum(1)
        
        
        a1, a2 = np.where(Closest[loc])
        I_update[a1, a2, :] = I_temp[a1, a2, :]
        pos_update[a1, a2, 0] = pos_temp[a1, a2, 0]
        pos_update[a1, a2, 1] = pos_temp[a1, a2, 1]       
        
        # set params to 0 for disappearing emitter        
        pos_update[loc,j2] = 0
        I_update[loc,j2,:] = 0
        
        # calculate the acceptance rate
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState_update, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState_update[:,:,None], backgroundParams, smpIndices)

        PR = np.exp(prior_new + ll_new - prior_prev - ll_prev)*(self.prior_func.roisize + 2*self.prior_func.border)**2/(self.prior_func.mu_k/emitterState_update.sum(1))
        u = np.random.uniform(0,1,n_eval)
                
        q = np.zeros(n_eval)
        q[loc] = gauss1d(u2, 0, self.sigma_PSF) * gauss1d(u3, 0, self.sigma_PSF) * gauss1d(u4, 0, self.sigma_PSF*self.z_rescale) * beta.pdf(u1,1,1)
        I_tot = (I[loc,:,:]*Closest[loc,:,None]).sum(1)               # combined intensity of all except the emitter turning off

        det_jac = np.zeros(n_eval)
        det_jac[loc] = (I_tot / (( 1 - u1)**(Np[loc]*2 + 1))[:,None]).prod(1)
        
        A = np.zeros(n_eval)
        A[loc] = PR[loc] * self.p_split/self.p_merge * q[loc] / det_jac[loc]
                
        failed_updates = (A < u) + np.isnan(A)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]
        I_update[failed_updates] = I[failed_updates]
        emitterState_update[failed_updates] = emitterState[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState_update, ~failed_updates, ll_new, prior_new

class Birth:
    def __init__(self, sigma_PSF, prior_func, LL_func, p_birth, p_death, z_range):
        self.sigma_PSF = sigma_PSF
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.p_birth = p_birth
        self.p_death = p_death
        self.z_range = z_range
        
    def residual_image(self, pos, I, backgroundParams, emitterState, loc):
        samples = self.LL_func.samples
        ev = self.LL_func.ExpectedValue(pos, I*emitterState[:,:,None])
        diff = samples[loc] - ev - backgroundParams[:,None,None,None]     
        diff[diff < 0] = 0
        return diff     
    
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        modelParams = emitterState.sum(1)
        n_eval, km, numFrames = I.shape
        
        # evaluate where modelParams < k_max
        births = len(modelParams[modelParams < k_max])
        if births == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev 
        
        loc = np.where(modelParams < k_max)[0]         # birthable sets
        j1 = np.zeros(len(loc), dtype = int)
        emitterState_update = np.copy(emitterState)

        for i in range(births):               # not i in loc
            j1[i] = np.arange(0, k_max)[~emitterState[loc[i]]][0]
            emitterState_update[loc[i], j1[i]] = True

        # construct residual image and select a pixel to generate a new emitter
        residual = self.residual_image(pos[loc], I[loc], backgroundParams[loc], emitterState[loc], loc)

        # multi-frame, take product of the frames to find birth probabilities
        residual = residual.prod(1)
        x_choice = np.zeros(births, dtype = int)
        y_choice = np.zeros(births, dtype = int)
        for i in range(births):    
            if residual[i,:,:].sum() == 0:
                residual[i,:,:] = 1
            
            x_choice[i] = np.random.choice(np.arange(np.size(residual,1)), p = residual[i,:,:].sum(1)/residual[i,:,:].sum())
            p2 = residual[i, x_choice[i], :]/residual[i,x_choice[i],:].sum()
            y_choice[i] = np.random.choice(np.arange(np.size(residual,2)), p = p2)
            
        # uniformly sample a position from within the pixel
        x_new = np.random.uniform(x_choice - .5, x_choice + .5)
        y_new = np.random.uniform(y_choice - .5, y_choice + .5)
        
        # added z for 3d move, uniformly sampled over 3d psf range
        z_new = np.random.uniform(self.z_range[0], self.z_range[1], size = births)
        u4 = 1 / (self.z_range[1] - self.z_range[0])
        
        
        pos_update = np.copy(pos)
        I_update = np.copy(I)
        pos_update[loc,j1,0] = x_new
        pos_update[loc,j1,1] = y_new
        pos_update[loc,j1,2] = z_new
        
        I_update[loc,j1,:] = np.random.choice(np.arange(len(self.prior_func.I_PDF)), size = [births, numFrames], p = self.prior_func.I_PDF)
        
        # calculate the acceptance rate
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState_update, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState_update[:,:,None], backgroundParams, smpIndices)
        
        # note: this assumes a square ROI
        PR = np.exp(ll_new - ll_prev + prior_new - prior_prev)
        u = np.random.uniform(0,1,n_eval)
        res_norm = residual/residual.sum(1).sum(1)[:,None,None]
        
        # added probability of sampling z position
        q = np.zeros(n_eval)
        q[loc] = res_norm[np.arange(births),x_choice,y_choice] * u4

        A = np.zeros(n_eval)
        A[loc] = PR[loc] * self.p_death/self.p_birth / q[loc]

        failed_updates = (A < u) + np.isnan(A)        # where posterior ratio < random number AND where posterior ratio isNaN
        pos_update[failed_updates] = pos[failed_updates]
        I_update[failed_updates] = I[failed_updates]
        emitterState_update[failed_updates] = emitterState[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos_update, I_update, backgroundParams, emitterState_update, ~failed_updates, ll_new, prior_new

class Death:
    def __init__(self, sigma_PSF, prior_func, LL_func, p_death, p_birth):
        self.sigma_PSF = sigma_PSF
        self.prior_func = prior_func
        self.LL_func = LL_func
        self.p_birth = p_birth
        self.p_death = p_death 
        
    def calc(self, pos, I, backgroundParams, emitterState, k_max, smpIndices, ll_prev, prior_prev):
        modelParams = emitterState.sum(1)
        n_eval = len(modelParams)
        
        # evaluate where modelParams < k_max
        deaths = len(modelParams[modelParams > 0])
        if deaths == 0:
            return pos, I, backgroundParams, emitterState, np.zeros(n_eval, dtype=bool), ll_prev, prior_prev 

        loc = np.where(modelParams > 0)[0]
        j1 = np.zeros(deaths, dtype = int)
        emitterState_update = np.copy(emitterState)
        for i in range(deaths):
            j1[i] = np.random.choice(np.arange(0, k_max)[emitterState[loc[i]]])     # pick elem for death, index j1 turns off
            emitterState_update[loc[i],j1[i]] = False
        

        pos_update = np.copy(pos)
        I_update = np.copy(I)
        pos_update[loc,j1,0] = 0
        pos_update[loc,j1,1] = 0
        pos_update[loc,j1,2] = 0
        I_update[loc,j1,:] = 0
        
        # calculate the acceptance rate
        prior_new = self.prior_func.calc(pos_update, I_update, backgroundParams, emitterState_update, smpIndices)
        ll_new = self.LL_func.Likelihood(pos_update, I_update*emitterState_update[:,:,None], backgroundParams, smpIndices)


        PR = np.exp(ll_new - ll_prev + prior_new - prior_prev)
        u = np.random.uniform(0,1,n_eval)

        A = np.zeros(n_eval)

        A[loc] = PR[loc] * self.p_birth/self.p_death
                
        failed_updates = (A < u) + np.isnan(A)        # where posterior ratio < random number AND where posterior ratio isNaN
        emitterState_update[failed_updates] = emitterState[failed_updates]
        ll_new[failed_updates] = ll_prev[failed_updates]
        prior_new[failed_updates] = prior_prev[failed_updates]
        return pos, I, backgroundParams, emitterState_update, ~failed_updates, ll_new, prior_new
    
    
    
def rjmcmc_3d_localization(jump_list, jump_prob, n_eval, numFrames,
                     pos_chain_0, I_chain_0, bg_chain_0, state_chain_0,
                     smpIndices, 
                     k_max = 10, chain_length=10000, ll_func=[], save_ev = False, roisize=20, print_preamble = ''):
    
    # number of datasets, max number of emitters, number of dimensions (xyzI = 4, xyI = 3), chain length
    a1, a2 = np.where(state_chain_0)
    pos_chain = np.zeros([n_eval, k_max, 3, chain_length])
    I_chain = np.zeros([n_eval, k_max, numFrames, chain_length])
    pos_chain[a1,a2,:,0] = pos_chain_0[a1,a2,:]
    I_chain[a1, a2, :, 0] = I_chain_0[a1, a2, :]

    # number of datasets, number of dimensions (bg, ax, ay), chain length
    bg_chain = np.zeros([n_eval, chain_length])
    bg_chain[:,0] = bg_chain_0
    
    # number of datasets, max emitters
    state_chain = np.zeros([n_eval, k_max, chain_length], dtype = bool)
    state_chain[:,:,0] = state_chain_0

    acceptance = np.zeros([n_eval, len(jump_list), chain_length], dtype = bool)
    jump_tracker = np.zeros([chain_length], dtype=int)
    
    ll_current = jump_list[0].LL_func.Likelihood(pos_chain_0, I_chain_0 * state_chain_0[:,:,None], bg_chain_0, smpIndices)
    prior_current = jump_list[0].prior_func.calc(pos_chain_0, I_chain_0, bg_chain_0, state_chain_0, smpIndices)
    
    ll_chain = np.zeros([n_eval, chain_length])
    prior_chain = np.zeros([n_eval, chain_length])
    ll_chain[:,0] = ll_current
    prior_chain[:,0] = prior_current
    if save_ev:
        ev_chain = np.zeros([n_eval, roisize, roisize, chain_length])
        ev_chain[:,:,:,0] = (ll_func.ExpectedValue(pos_chain[:,:,:,0], I_chain[:,:,:,0]*state_chain[:,:,None,0]) + bg_chain[:,0,None,None,None]).transpose([0,2,3,1]).squeeze()
    
    for i in range(chain_length-1):
        # jump selection
        a = np.random.choice(np.arange(len(jump_list)), p = jump_prob)      # for logging the selected jumps and acceptance rates
        jump = jump_list[a]
        jump_tracker[i] = a
        # execute the jump
        pos_chain[:,:,:,i+1], I_chain[:,:,:,i+1], bg_chain[:,i+1], state_chain[:,:,i+1], acceptance[:,a,i], ll_chain[:,i+1], prior_chain[:,i+1] = jump.calc(pos_chain[:,:,:,i], I_chain[:,:,:,i], bg_chain[:,i], state_chain[:,:,i], k_max, smpIndices, ll_chain[:,i], prior_chain[:,i])

        if save_ev:
            ev_chain[:,:,:,i+1] = (ll_func.ExpectedValue(pos_chain[:,:,:,i+1], I_chain[:,:,:,i+1]*state_chain[:,:,None,i+1]) + bg_chain[:,i+1,None,None,None]).transpose([0,2,3,1]).squeeze()

        
        if np.mod(i,200) == 0:           # print an update every x iterations
            print(print_preamble + f'iteration: {i}')# | moves: {moves_accepted}/{moves} | splits: {splits_accepted}/{splits} | merges: {merges_accepted}/{merges} | births: {births_accepted}/{births} | deaths: {deaths_accepted}/{deaths} ')

    if save_ev:
        return pos_chain, I_chain, bg_chain, state_chain, acceptance, jump_tracker, ll_chain, prior_chain, ev_chain
    else:
        return pos_chain, I_chain, bg_chain, state_chain, acceptance, jump_tracker, ll_chain, prior_chain
    


class MCMC:
    def __init__(self):
        self.name = 'mcmc_parameter/result_object'
        
class RJMCMC:
    def __init__(self):
        self.name = 'rjmcmc_parameter/result_object'
        

class RJMCMC_Object:
    def __init__(self, ctx, psf, smp, z_range, sigma_PSF, z_rescale = 1):
        self.ctx = ctx
        self.psf = psf
        self.smp = smp
        self.n_eval = smp.shape[0]
        self.numFrames = smp.shape[1]
        self.roisize = smp.shape[2]
        self.z_range = z_range
        self.sigma_PSF = sigma_PSF
        self.smpIndices = np.arange(self.n_eval, dtype = int)
        self.z_rescale = z_rescale
        
    def set_priors(self, border = 4, k_max = 10, mu_k = 5, bg_offset_range = [3, 30], 
                   mu_I = 2000, sigma_I = 150, I_taper = 500, frac = 40, 
                   I_PDF_from_SMLM = False):
        '''
        Sets all the priors: 
            - emitter count from a Poisson distribution
            - background offset from uniform distribution
            - emitter intensity either from SMLM kernel density
                fit or from user defined gaussian
            - emitter position from uniform distribution
        and makes the prior + likelihood calculators.
        '''
        self.k_max = k_max
        self.mu_k = mu_k
        self.border = border
        self.k_PDF = poisson._pmf(np.arange(0, k_max + 1,), mu_k)
        self.bg_PDF = np.zeros(10000)
        self.bg_PDF[bg_offset_range[0]: bg_offset_range[1]] = 1
        self.bg_PDF /= self.bg_PDF.sum()
        if I_PDF_from_SMLM:
            #KernelDensity( bandwidth = 75, kernel = 'gaussian')
            return 0
        else:
            #sI = 500
            #self.I_PDF = gauss1d(np.arange(30000), mu_I, sI)
            #self.I_PDF[int(mu_I + 3*sI)::] = 0
            self.I_PDF = gauss1d(np.arange(30000), mu_I, sigma_I)
            self.I_PDF[int(mu_I + 3*sigma_I)::] = 0
            self.I_PDF[I_taper:np.where(self.I_PDF > max(self.I_PDF)/frac)[0][0]] = max(self.I_PDF)/frac
            self.I_PDF[0:I_taper] = np.linspace(0,max(self.I_PDF)/frac, I_taper)
            self.I_PDF /= self.I_PDF.sum()
        self.p_calc = Prior_calculator(self.n_eval, prior_I, self.I_PDF, prior_bg, self.bg_PDF,
                                       prior_xyz, self.roisize, self.border, self.z_range[0], 
                                       self.z_range[1], prior_k, self.k_PDF, self.mu_k, 
                                       self.k_max)
        
        self.intensityPDF = PDFInfo(self.I_PDF[100:3000],100, 3000)    
        self.mcl = MCLocalizer(self.k_max, self.numFrames, self.psf, self.intensityPDF, self.ctx)
        self.mcl.SetSamples(self.smp, self.smp*0)      

    def set_rjmcmc_moves(self, move_prob = np.array((3,3,3,1,1,1,1,1,1))/15, 
                         move_prob_2 = np.array((4,4,4,0,0,1.5,1.5,.5,.5))/16, 
                         p_split = .5, p_gsplit = .5, p_birth = .5,
                         sigma_x = 0.05, sigma_y = 0.05, sigma_z = 0.01,
                         sigma_I = 5, sigma_bg = 1,
                         rjmcmc_len = 5000, rjmcmc_burn = 3000):
        '''
        Generates all the moves and required parameters using 
        bamf's probabilities for move selection in burn-in
        and post-burn-in phase.
        '''
        
        self.rjmcmc = RJMCMC()
        self.move_prob = move_prob
        self.move_prob_2 = move_prob_2
        self.rjmcmc_len = rjmcmc_len
        self.rjmcmc_burn = rjmcmc_burn
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.sigma_I = sigma_I
        self.sigma_z = sigma_z
        self.sigma_bg = sigma_bg
        
        self.split_move = Split(self.sigma_PSF, self.p_calc, self.mcl, p_split, 1-p_split, z_rescale = self.z_rescale)
        self.merge_move = Merge(self.sigma_PSF, self.p_calc, self.mcl, 1-p_split, p_split, z_rescale = self.z_rescale)
        self.gsplit_move = Generalized_Split(self.sigma_PSF, self.p_calc, self.mcl, p_gsplit, 1-p_gsplit, z_rescale = self.z_rescale)
        self.gmerge_move = Generalized_Merge(self.sigma_PSF, self.p_calc, self.mcl, 1-p_gsplit, p_gsplit, z_rescale = self.z_rescale)
        self.birth_move = Birth(self.sigma_PSF, self.p_calc, self.mcl, p_birth, 1-p_birth, z_range = self.z_range)
        self.death_move = Death(self.sigma_PSF, self.p_calc, self.mcl, 1-p_birth, p_birth)
        
        self.single_move = Single_Emitter_Move(sigma_x, sigma_y, sigma_z, sigma_I, self.p_calc, self.mcl)
        self.bg_move = Background_Move(sigma_bg, 1e-9, 1e-9, self.p_calc, self.mcl)
        self.group_move_uncon = Group_Move_Unconserved(self.sigma_PSF, self.sigma_x, self.sigma_y, 
                                                       self.sigma_z, self.sigma_I, self.p_calc, self.mcl)
        
        self.split_move.name = 'split'
        self.merge_move.name = 'merge'
        self.gsplit_move.name = 'gsplit'
        self.gmerge_move.name = 'gmerge'
        self.birth_move.name = 'birth'
        self.death_move.name = 'death'
        self.single_move.name = 'single'
        self.bg_move.name = 'bg'
        self.group_move_uncon.name = 'group_uncon'
    
        self.move_list = [self.group_move_uncon, self.single_move, self.bg_move, self.split_move, 
                          self.merge_move, self.gsplit_move, self.gmerge_move, self.birth_move, 
                          self.death_move]

        
    def set_mcmc_moves(self, move_prob_mcmc = [.4, .4, .2], mcmc_len = 3000, mcmc_burn = 0,
                       sigma_x = 0.05, sigma_y = 0.05, sigma_z = 0.01,
                       sigma_I = 5, sigma_bg = 1):
        '''
        Generates separate moves for the MCMC portion of
        the algorithm.
        '''
        self.mcmc = MCMC()
        self.mcmc.sigma_x = sigma_x
        self.mcmc.sigma_y = sigma_y
        self.mcmc.sigma_z = sigma_z
        self.mcmc.sigma_I = sigma_I
        self.mcmc.sigma_z = sigma_z
        self.mcmc.sigma_bg = sigma_bg
        
        self.mcmc.single_move = Single_Emitter_Move(sigma_x, sigma_y, sigma_z, sigma_I, self.p_calc, self.mcl)
        self.mcmc.bg_move = Background_Move(sigma_bg, 1e-9, 1e-9, self.p_calc, self.mcl)
        self.mcmc.group_move_uncon = Group_Move_Unconserved(self.sigma_PSF, self.sigma_x, self.sigma_y, 
                                                       self.sigma_z, self.sigma_I, self.p_calc, self.mcl)
        
        self.mcmc.move_list = [self.mcmc.group_move_uncon, self.mcmc.single_move, self.mcmc.bg_move]
        self.mcmc.move_prob = move_prob_mcmc
        
        self.mcmc_len = mcmc_len
        self.mcmc_burn = mcmc_burn

    def run_rjmcmc(self, save_ev = False, print_preamble = '', **kwargs):
        '''
        runs RJMCMC on the data using a burn-in portion with high model 
        move selection probability and a post burn-in portion which
        uses a different move selection probability
        '''
        # randomly draw an initial localization from the priors, turning only 1 emitter on in each frame
        pos_chain_0 = np.random.uniform(0, self.roisize, size = [self.n_eval, self.k_max, 3])
        pos_chain_0[:,:,-1] = np.random.uniform(self.z_range[0], self.z_range[1], [self.n_eval, self.k_max])
        
        # initialize from I pdf and then rescale according to d and emitter z
        I_chain_0 = np.random.choice(np.arange(len(self.I_PDF)), p = self.I_PDF, size = [self.n_eval, self.k_max, self.numFrames])

        bg_chain_0 = np.random.choice(np.arange(len(self.bg_PDF)), p = self.bg_PDF, size = [self.n_eval])
        
        state_chain_0 = np.zeros([self.n_eval, self.k_max], dtype=bool)
        state_chain_0[:,0] = True
        pos_chain_0 *= state_chain_0[:,:,None]
        I_chain_0 *= state_chain_0[:,:,None]

        if len(kwargs) > 0:
            try:
                pos_chain_0 = kwargs['pos_0']
            except:
                pass
            try:
                state_chain_0 = kwargs['state_0']
            except:
                pass  
            try:
                I_chain_0 = kwargs['I_0']
            except:
                pass
            try:
                bg_chain_0 = kwargs['bg_0']
            except:
                pass
        
        # run RJMCMC Burn-in
        print('running burn-in portion of RJMCMC')
        if save_ev:
                self.rjmcmc.pos_chain_burn, self.rjmcmc.I_chain_burn, self.rjmcmc.bg_chain_burn, self.rjmcmc.state_chain_burn, self.rjmcmc.acceptance_burn, self.rjmcmc.jump_tracker_burn, self.rjmcmc.ll_chain_burn, self.rjmcmc.prior_chain_burn, self.rjmcmc.ev_chain_burn = rjmcmc_3d_localization(self.move_list, self.move_prob, self.n_eval, self.numFrames,
                                                               pos_chain_0, I_chain_0, bg_chain_0, state_chain_0,
                                                               self.smpIndices, 
                                                               k_max = self.k_max, chain_length=self.rjmcmc_burn, ll_func = self.mcl, save_ev = True, roisize = self.roisize, print_preamble = print_preamble)
        else:
            self.rjmcmc.pos_chain_burn, self.rjmcmc.I_chain_burn, self.rjmcmc.bg_chain_burn, self.rjmcmc.state_chain_burn, self.rjmcmc.acceptance_burn, self.rjmcmc.jump_tracker_burn, self.rjmcmc.ll_chain_burn, self.rjmcmc.prior_chain_burn = rjmcmc_3d_localization(self.move_list, self.move_prob, self.n_eval, self.numFrames,
                                                               pos_chain_0, I_chain_0, bg_chain_0, state_chain_0,
                                                               self.smpIndices, 
                                                               k_max = self.k_max, chain_length=self.rjmcmc_burn, ll_func = self.mcl, roisize=self.roisize, print_preamble = print_preamble)

        # run RJMCMC post burn
        print('running post burn-in RJMCMC')
        if save_ev:
            self.rjmcmc.pos_chain, self.rjmcmc.I_chain, self.rjmcmc.bg_chain, self.rjmcmc.state_chain, self.rjmcmc.acceptance, self.rjmcmc.jump_tracker, self.rjmcmc.ll_chain, self.rjmcmc.prior_chain, self.rjmcmc.ev_chain =      rjmcmc_3d_localization(self.move_list, self.move_prob_2, self.n_eval, self.numFrames,
                                                               self.rjmcmc.pos_chain_burn[:,:,:,-1], self.rjmcmc.I_chain_burn[:,:,:,-1],
                                                               self.rjmcmc.bg_chain_burn[:,-1], self.rjmcmc.state_chain_burn[:,:,-1],
                                                               self.smpIndices, 
                                                               k_max = self.k_max, chain_length=self.rjmcmc_len - self.rjmcmc_burn, ll_func = self.mcl, save_ev = True, roisize = self.roisize, print_preamble = print_preamble)
        else:
            self.rjmcmc.pos_chain, self.rjmcmc.I_chain, self.rjmcmc.bg_chain, self.rjmcmc.state_chain, self.rjmcmc.acceptance, self.rjmcmc.jump_tracker, self.rjmcmc.ll_chain, self.rjmcmc.prior_chain =      rjmcmc_3d_localization(self.move_list, self.move_prob_2, self.n_eval, self.numFrames,
                                                               self.rjmcmc.pos_chain_burn[:,:,:,-1], self.rjmcmc.I_chain_burn[:,:,:,-1],
                                                               self.rjmcmc.bg_chain_burn[:,-1], self.rjmcmc.state_chain_burn[:,:,-1],
                                                               self.smpIndices, 
                                                               k_max = self.k_max, chain_length=self.rjmcmc_len - self.rjmcmc_burn, ll_func = self.mcl, roisize=self.roisize, print_preamble = print_preamble)           

    def select_MAPN(self):
        '''
        Select the optimal model: number of emitters with the highest maximum a posteriori probability
        '''
        print('finding MAP number of emitters')        
        self.found_emitters = np.zeros(self.n_eval, dtype = int)
        self.mcmc.pos_0 = np.zeros([self.n_eval, self.k_max, 3])
        self.mcmc.I_0 = np.zeros([self.n_eval, self.k_max, self.numFrames])
        self.mcmc.bg_0 = np.zeros(self.n_eval)
        self.mcmc.state_0 = np.zeros([self.n_eval, self.k_max], dtype = bool)

        for i in range(self.n_eval):
            unique, counts = np.unique(self.rjmcmc.state_chain.sum(1)[i], return_counts = True)
            self.found_emitters[i] = unique[counts == max(counts)]
            self.mcmc.pos_0[i,:,:]  = self.rjmcmc.pos_chain[i,:,:,self.rjmcmc.state_chain.sum(1)[i] == self.found_emitters[i]].transpose((1,2,0))[:,:,-1]
            self.mcmc.I_0[i,:,:]    = self.rjmcmc.I_chain[i,:,:,self.rjmcmc.state_chain.sum(1)[i] == self.found_emitters[i]].transpose((1,2,0))[:,:,-1]
            self.mcmc.bg_0[i]       = self.rjmcmc.bg_chain[i, self.rjmcmc.state_chain.sum(1)[i] == self.found_emitters[i]][-1]
            self.mcmc.state_0[i,:]  = self.rjmcmc.state_chain[i, :, self.rjmcmc.state_chain.sum(1)[i] == self.found_emitters[i]].transpose()[:,-1]
           
         
    def run_mcmc(self, save_ev = False, print_preamble = ''):
        '''
        runs MCMC on the most likely model of number of emitters
        '''
        print('running MCMC')
        if save_ev:
            self.mcmc.pos_chain, self.mcmc.I_chain, self.mcmc.bg_chain, self.mcmc.state_chain, self.mcmc.acceptance, self.mcmc.jump_tracker, self.mcmc.ll_chain, self.mcmc.prior_chain, self.mcmc.ev_chain =      rjmcmc_3d_localization(self.mcmc.move_list, self.mcmc.move_prob, self.n_eval, self.numFrames,
                                                               self.mcmc.pos_0, self.mcmc.I_0,
                                                               self.mcmc.bg_0, self.mcmc.state_0,
                                                               self.smpIndices, 
                                                               k_max = self.k_max, chain_length=self.mcmc_len, ll_func = self.mcl, save_ev = True, roisize=self.roisize, print_preamble = print_preamble)
        else:
            self.mcmc.pos_chain, self.mcmc.I_chain, self.mcmc.bg_chain, self.mcmc.state_chain, self.mcmc.acceptance, self.mcmc.jump_tracker, self.mcmc.ll_chain, self.mcmc.prior_chain =      rjmcmc_3d_localization(self.mcmc.move_list, self.mcmc.move_prob, self.n_eval, self.numFrames,
                                                               self.mcmc.pos_0, self.mcmc.I_0,
                                                               self.mcmc.bg_0, self.mcmc.state_0,
                                                               self.smpIndices, 
                                                               k_max = self.k_max, chain_length=self.mcmc_len, ll_func = self.mcl, roisize=self.roisize, print_preamble = print_preamble)
        

    
