import numpy as np
from Classical.HMM.HMM import HMM, HMM_EMFiter_BW

mat_t = np.mat([[0.01, 0.98, 0.01], [0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
mat_o = np.mat([[0.96, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.96, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.96, 0.01, 0.01]])
# This emission matrix is shape [n_states, n_obs], the transpose of my HMM class
initial_state = np.mat([[0.98, 0.01, 0.01]])


def get_random_sequence(length: int):
    seq = []
    state_dis = initial_state
    for i in range(length):
        state_dis = np.matmul(state_dis, mat_t)
        cur_state = np.random.choice(3, 1, p=state_dis.getA()[0, :])[0]
        cur_ob = np.random.choice(5, 1, p=mat_o.getA()[cur_state, :])[0]
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

def test_forward_backward():
    pss = [get_state_probability([0, 2, 1, 1], i, exp_time=10000) for i in range(4)]
    pss = [ps/10000 for ps in pss]
    print(pss, sep='\n')


    hmm = HMM(3, 5, initial_state.T, mat_t, mat_o.T)
    probs, _ = hmm.forward_backward([0, 2, 1, 1])
    print(probs, sep='\n')

# test_forward_backward()



hmm1 = HMM(3, 5)
fiter = HMM_EMFiter_BW(hmm1)
num_trues = []
for i in range(100):
    print("EM Fit:", i)
    ys = get_random_sequence(60)[:, 1]
    print(ys)
    if i % 1 == 0:
        num_true = 0
        for j in range(400):
            test_ys = get_random_sequence(8)[:, 1]
            pred_probs = hmm1.predict(test_ys[:-1])[:, 0]
            pred = np.argmax(pred_probs)
            if pred == test_ys[-1]:
                num_true += 1
        num_trues.append(num_true)
        print("Num true:", num_true)

    fiter.em_one_seq(ys)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(hmm1.initial_state, hmm1.mat_t, hmm1.mat_o, sep="\n")

import matplotlib.pyplot as plt
plt.plot(num_trues)
plt.show()