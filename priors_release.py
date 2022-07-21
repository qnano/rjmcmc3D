import numpy as np

def gauss1d(x, x_pos=0,sigma=1):
    return np.exp((-(x-x_pos)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


def prior_k(emitterState, k_PDF, k_max):

    p_k = np.zeros(np.shape(emitterState)[0])
    K = np.array(emitterState.sum(1), dtype = int)
    k_PDF2 = np.concatenate([np.copy(k_PDF),np.array([0])],axis=0)
    K[(K < 1)] = 0
    K[(K > k_max)] = 0
    np.putmask(p_k, (K >= 0) & (K <= k_max), k_PDF2[K])
    return p_k

def prior_I(I, emitterState, I_PDF):

    I2 = I.astype(int)
    p_I = np.zeros(np.shape(I2))
    es2 = np.zeros(I.shape)
    es2[:,:,:] = ~emitterState[:,:,None]
    I2[(I2 > len(I_PDF))] = -1
    I2[(I2 < 0)] = 0
    I2[(I2 >= len(I_PDF))] = 0
    np.putmask(p_I, (I2 < len(I_PDF)) & (I2 >= 0), I_PDF[I2])
    np.putmask(p_I, es2, 1)
    return p_I.prod(2)

def prior_bg(bg_in, bg_PDF):

    bg = bg_in.astype(int)
    p_bg= np.zeros(np.shape(bg))
    bg[bg < 0] = -1
    np.putmask(p_bg, bg >= 0, bg_PDF[bg])
    return p_bg
    
def prior_xyz(pos, emitterState, roisize, border, zmin, zmax):
    x = pos[:,:,0]
    y = pos[:,:,1]
    z = pos[:,:,2]
    p_xy = np.zeros(np.shape(x))
    np.putmask(p_xy,(x > -border) & (x < roisize + border) & (y > -border) & (y < roisize + border) & (z > zmin) & (z < zmax), 1/(roisize + 2*border)**2/(zmax - zmin))  
    np.putmask(p_xy, ~emitterState, 1)      
    return p_xy

class Prior_calculator:
    def __init__(self, n_eval, pI_func, I_PDF, pbg_func, bg_PDF, pxy_func, roisize, border, zmin, zmax, pk_func, k_PDF, mu_k, k_max):
        self.pI_func = pI_func
        self.I_PDF = I_PDF
        self.pbg_func = pbg_func
        self.bg_PDF = bg_PDF
        self.pxy_func = pxy_func
        self.roisize = roisize
        self.border = border
        self.zmin = zmin
        self.zmax = zmax
        self.pk_func = pk_func
        self.k_PDF = k_PDF
        self.mu_k = mu_k
        self.k_max = k_max
        self.n_eval = n_eval
        self.a = 0
        self.z0 = 0
    def calc(self, pos, I, backgroundParams, emitterState, loc = []):
        pi_o = np.log(self.pI_func(I, emitterState, self.I_PDF))
        pbg_o = np.log(self.pbg_func(backgroundParams, self.bg_PDF))
        pxy_o = np.log(self.pxy_func(pos, emitterState, self.roisize, self.border, self.zmin, self.zmax))
        pk_o = np.log(self.pk_func(emitterState, self.k_PDF, self.k_max))
        return (pi_o + pxy_o).sum(1) + pbg_o + pk_o 
    def prior_mat(self, pos, I, backgroundParams, emitterState):
        priors = np.zeros([pos.shape[0], 4],dtype=float)
        # pI, pxy, pbg, pk
        priors[:,0] = np.log(self.pI_func(I, emitterState, self.I_PDF)).sum(1)
        priors[:,1] = np.log(self.pxy_func(pos, emitterState, self.roisize, self.border, self.zmin, self.zmax)).sum(1)
        priors[:,2] = np.log(self.pbg_func(backgroundParams, self.bg_PDF))
        priors[:,3] = np.log(self.pk_func(emitterState, self.k_PDF, self.k_max))
        return priors
