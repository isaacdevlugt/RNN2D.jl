import numpy as np

# Some code directly from https://github.com/emerali/rand_wvfn_sampler/blob/master/data_gen_py.ipynb

def generate_hilbert_space(size):
    dim = np.arange(2 ** size)
    space = (((dim[:, None] & (1 << np.arange(size)))) > 0)[:, ::-1]
    space = space.astype(int)
    return space

def get_samples_from_psi_indices(indices, N):
    return (((indices[:, None] & (1 << np.arange(N)))) > 0)[:, ::-1].astype(int)

def gen_samples(num_samples, N, psi):
    probs = psi * psi
    indices = np.random.choice(len(probs), size=num_samples, p=probs)
    return indices, get_samples_from_psi_indices(indices, N)

def gen_data(num_samples):
    N = 9
    size = 2 ** N
    vis = generate_hilbert_space(N)

    psi = np.loadtxt("true_psi_L=3_Rb=1.2_delta=1.12.dat")[:,0] # real part
    _, samples = gen_samples(num_samples, N, psi)

    np.savetxt("ED_L=3_Rb=1.2_delta=1.12_samples.dat", samples, fmt='%.0f')

gen_data(100000)