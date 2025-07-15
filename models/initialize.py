import numpy as np

def gauss_initial(grid, max_value, mu, sigma=None, devs=None):
    if sigma is not None:
        if not np.allclose(sigma, sigma.T):
            raise ValueError("Only symmetric variance-covariance-matrices are accepted.")
    elif devs is not None:
        sigma = np.diag(devs)**2
    else:
        raise ValueError("No values were given to calculate the variance-covariance matrix.")

    mu = np.asarray(mu)
    inv = np.linalg.inv(sigma)
    coord = [np.linspace(0,1,grid[i]) - mu[i] for i in range(len(grid))]
    coord = np.asarray(np.meshgrid(*coord, indexing='ij'))
    densDist = max_value*np.exp(-0.5*np.einsum('i...,...i', coord, np.einsum('ij,i...', inv, coord)))

    if densDist.max() > 1. or densDist.min() < 0.:
        print("The choice of the parameters led to physically meaningless results.")

    return densDist

def sinusoid_initial(grid, frequencies, amplitude=0.5, intercept=0.5, shift=0.):
    nu = np.diag(list(frequencies))
    coord = [np.linspace(0,1,grid[i]) for i in range(len(grid))]
    coord = np.asarray(np.meshgrid(*coord, indexing='ij'))
    densDist = amplitude*np.cos(2*np.pi*(np.sum(np.einsum('ij,i...', nu, coord), axis=len(grid)) + shift)) + intercept
    if densDist.max() > 1. or densDist.min() < 0.:
        print("The choice of the parameters led to physically meaningless results.")
    return densDist

def gradient_initial(grid, *points):
    points = np.asarray(points)
    conc = points[:,2]
    M = np.ones(points.shape)
    M[:,:2] = points[:,:2]
    coeff, *_ = np.linalg.lstsq(M, conc, rcond=None)
    A = np.diag(coeff[:2])
    coord = [np.linspace(0,1,grid[i]) for i in range(len(grid))]
    coord = np.asarray(np.meshgrid(*coord, indexing='ij'))
    densDist = np.sum(np.einsum('ij,i...', A, coord), axis=np.size(coeff)-1) + coeff[2]
    densDist = np.clip(densDist, 0., 1.)
    return densDist
    
def patches(N, num_patches=8, patch_size=8):
    u = np.ones((N, N))
    v = np.zeros((N, N))

    for _ in range(num_patches):
        i = np.random.randint(0, N - patch_size)
        j = np.random.randint(0, N - patch_size)

        u[i:i+patch_size, j:j+patch_size] = 0.5 + 0.1*np.random.rand(patch_size, patch_size)
        v[i:i+patch_size, j:j+patch_size] = 0.25 + 0.1*np.random.rand(patch_size, patch_size)

    return u, v

def initialize(N, pattern='patches'):
    grid = (N, N)
    if pattern == 'random':
        u = np.random.rand(*grid)
        v = np.random.rand(*grid)
    elif pattern == 'gaussian':
        u = np.ones(grid)
        v = gauss_initial(grid, max_value=1.0, mu=(0.5, 0.5), devs=(0.1, 0.1))
    elif pattern == 'sinusoidal':
        u = np.ones(grid)
        v = sinusoid_initial(grid, frequencies=(4, 4))
    elif pattern == 'gradient':
        u = gradient_initial(grid, (0, 0, 0), (1, 1, 1))
        v = gradient_initial(grid, (0, 0, 1), (1, 1, 0))
    elif pattern == 'patches':
        u, v = patches(N)
    else:
        u = np.ones(grid)
        v = np.zeros(grid)
    return u, v