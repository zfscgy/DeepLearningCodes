import numpy as np
from Classical.HMM.HMM import HMM

mat_t = np.mat([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
mat_o = np.mat([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
initial_state = np.mat([[0.5, 0.3, 0.2]])


def get_random_sequence(length: int):
    seq = []
    state_dis = initial_state
    for i in range(length):
        state_dis = np.matmul(state_dis, mat_t)
        cur_state = np.random.choice(3, 1, p=state_dis.getA()[0, :])[0]
        cur_ob = np.random.choice(3, 1, p=mat_o.getA()[cur_state, :])[0]
        seq.append([cur_state, cur_ob])
        state_dis = np.zeros([3])
        state_dis[cur_state] = 1
    return np.array(seq)


test_seq = get_random_sequence(3)


def get_state_probability(ys, xi, exp_time=1000):
    length = len(ys)
    xs = [0, 0, 0]
    for i in range(exp_time):
        seq = get_random_sequence(length)
        if np.array_equal(seq[:, 1], ys):
            xs[seq[xi, 0]] += 1
    return np.array(xs)

pss = [get_state_probability([0, 2, 1, 1], i, exp_time=100000) for i in range(4)]
pss = [ps/sum(ps) for ps in pss]
print(pss, sep='\n')


hmm = HMM(3, 3, initial_state.T, mat_t, mat_o.T)
probs = hmm.forward_backward([0, 2, 1, 1])
print(probs, sep='\n')