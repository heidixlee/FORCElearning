# setup!
import numpy as np

import matplotlib

import math


# allows for plots to be saved without opening another window
matplotlib.use("Agg")

import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


# parameters 

# number of neuronal units
N = 500

# probability of connections (sparse)
p = 0.1

# scale of activity (paper states that it should be greater than 1)
g = 1.3

# speed at which RLS is enforced
alpha = 1.0

# every change in x is dt/tau
dt = 1.0
tau = 10.0

# number of steps
n_pre = 500 # steps to allow reservoir to settle into activity
n_training = 4000 # training
n_test = 900 # stimulation without training
n_steps = n_pre + n_training + n_test



# resevoir (chaotic and random and sparse)

# J = N by N array of all neuronal units and their connections
# connections are randomly chosen in a Gaussian distribution
J = rng.normal(size = (N, N))

# scale so that variance is 1/P*N
J = J * g / math.sqrt(p * N)

# mask array so that probability of connections is p
# randomizes N by N array. if value is greater than 1-p, sets it to true
mask = rng.random(size = (N, N)) < p

# apply mask to J, making connections sparse
J = J * mask

# feedback array J is randomly chosen and is scaled by the weight vector w later on
Jfb = rng.normal(size = (N))

# correlation matrix (Identity matrix / alpha) --> At first all connections are treated independently
P = np.identity(N) / alpha

# initialize other variables
w = np.zeros(N)
x = rng.normal(0, 0.1, N)
r = np.tanh(x)
z = 0.0


# target function
period = 120
omega = (2 * math.pi) / period

# target sine wave as a function of t
def target_func(t):
    f = (np.sin (omega * t) 
    + 0.5 * np.sin (2 * omega * t) 
    + (1/3) * np.sin (3 * omega * t) 
    + (0.25) * np.sin (4 * omega * t))

    # keeps the target functions relatively between -1 and 1
    return f / 1.5

# simulation run

# stores history of z, target, and w
zHist = np.zeros(n_steps)
targetHist = np.zeros(n_steps)
wHist = np.zeros(n_steps)

# for the amount of time steps
for i in range(n_steps): 
    # t = i in terms of the simulation time step
    t = i * dt

    # target function at time t is defined
    target = target_func(t)
    
 # 3 variables always being updated = x, r, z
   
    # internal state, updated by recurrent neural activity and feedback
    # J @ r = Jr (neuronal activity for each neuron unit)
    # Jfb * z = Jzbz (feedback weight for each neuron multiplied by the feedback information that will be sent to network)
    x = x + (dt/tau) * (-x + J @ r + Jfb * z)

    # neural activity 
    r = np.tanh(x)

    # z = combined sum of w (weights) and r (neuron activity)
    z = w @ r

    

    # if in time step training, execute RLS and FORCE
    if (n_pre < i < n_pre + n_training):

        # correlation for each neuron as a vector, how much r is "worth"
        Pr = P @ r       

        # novelty of the activity pattern, scalar value
        rPr = r @ Pr 

        # gain factor (starts close to 1 --> converges near 0 rapidly)
        k = 1 / (1 + rPr)

        P -= k * np.outer(Pr, Pr)

        # identify the current error
        error = z - target

        # update the feedback weight, based on gain factor, error, and the correlation matrix
        w -= k * error * Pr

    zHist[i] = z
    targetHist[i] = target
    wHist[i] = np.linalg.norm(w)



# 5. Plotting Data

test_err = np.sqrt(np.mean((zHist[-n_test:] - targetHist[-n_test:]) ** 2))
print(f"RMS error during test (learning off): {test_err:.4f}")
 
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
t_axis = np.arange(n_steps) * dt
 
axes[0].plot(t_axis, targetHist, 'k', label='target', lw=1)
axes[0].plot(t_axis, zHist, 'r', label='network output', lw=1)
axes[0].axvline(n_pre * dt, color='gray', ls='--')
axes[0].axvline((n_pre + n_training) * dt, color='gray', ls='--')
axes[0].set_title('Network output vs target (dashed lines = learning on / off)')
axes[0].legend()
 
axes[1].plot(t_axis, wHist)
axes[1].set_title('||w|| over time (readout weights converging)')
 
test_t = t_axis[n_pre + n_training:]
axes[2].plot(test_t, targetHist[n_pre + n_training:], 'k', label='target')
axes[2].plot(test_t, zHist[n_pre + n_training:], 'r', label='output (autonomous)')
axes[2].set_title('Test period (learning off) - zoomed in')
axes[2].legend()
 
plt.tight_layout()
plt.savefig('force_result.png', dpi=120)
print("Saved plot to force_result.png")
