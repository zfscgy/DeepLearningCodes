import numpy as np


class HMM:
    def __init__(self, n_states, n_obs,
                 initial_state: np.ndarray = None,
                 mat_t: np.ndarray = None,
                 mat_o: np.ndarray = None):
        """

        :param n_states:
        :param n_obs:
        :param initial_state: [n_states, 1]
        :param mat_t:  [n_states, n_states] mat_t[i,j] = P(X(t+1) = i|X(t) = j)
        :param mat_o: [n_observations, n_states] mat_o[i,j] = P(Ob(t)=j|X(t)=i)
        """
        self.n_states = n_states
        self.n_obs = n_obs
        if mat_t is None:
            self.mat_t = np.random.uniform(0, 1, [n_states, n_states])
            self.mat_t /= np.sum(self.mat_t, axis=0, keepdims=True)  # [1, n_states]
            # Transition matrix's column sum must be 1
        else:
            assert mat_t.shape == (n_states, n_states), "Transition matrix shape wrong"
            self.mat_t = np.array(mat_t)
        if mat_o is None:
            self.mat_o = np.random.uniform(0, 1, [n_obs, n_states])
            self.mat_o /= np.sum(self.mat_o, axis=0, keepdims=True)
        else:
            assert mat_o.shape == (n_obs, n_states), "Emission matrix shape wrong"
            self.mat_o = np.array(mat_o)
        if initial_state is None:
            self.initial_state = np.random.uniform(0, 1, [n_states, 1])
            self.initial_state /= np.sum(self.initial_state)
        else:
            assert initial_state.shape == (n_states, 1), "Initial state matrix shape wrong"
            self.initial_state = np.array(initial_state)

    def forward(self, ys):
        """

        :param ys:
        :return: [P(x0), P(x1,y1), P(x2,y1..y2) ... P(xn, y1..yn)] where x0 is initial_state
        """
        probs = [self.initial_state]
        for i in range(len(ys)):
            prob = self.mat_o[ys[i], :].reshape([self.n_states, 1]) * \
                np.array(np.matmul(self.mat_t.transpose(), probs[i]))
            probs.append(prob)
        return probs

    def backward(self, ys):
        """

        :param ys:
        :return: [P(y1...yn| x0), P(y2...yn| x1) ,... P(yn| xn-1), ones([n_states, 1])]
        """
        probs = [np.ones([self.n_states, 1])]
        for i in reversed(range(len(ys))):
            prob = np.matmul(self.mat_t, self.mat_o[ys[i], :].reshape([self.n_states, 1]) * probs[0])
            probs.insert(0, prob)
        return probs

    def forward_backward(self, ys):
        """

        :param ys:
        :return: An array with shape [length, n_states, 1] representing P(x_t, y1...T),
                 An Tuple contains forward results and backward results(in array format)
        """
        probs_x_ys = np.array(self.forward(ys))  # p(xt, y1...t) [y_len, n_states, 1]
        probs_ys_on_x = np.array(self.backward(ys))  # p(y_t+1...T|xt) [y_len, n_states, 1]
        probs = np.multiply(probs_x_ys[1:, :, :], probs_ys_on_x[1:, :, :])
        return probs, [probs_x_ys, probs_ys_on_x]

    def predict(self, ys):
        xT_prob = self.forward(ys)[-1]  # p(xt, yt) [n_states, 1]
        xT_prob /= sum(xT_prob)
        xTp1_prob = np.matmul(self.mat_t.transpose(), xT_prob)
        yTp1_prob = np.matmul(self.mat_o, xTp1_prob)
        return yTp1_prob


class HMM_EMFiter_BW:
    def __init__(self, hmm_model: HMM, eps=1e-8):
        self.hmm = hmm_model
        self.eps = eps

    def em_one_seq(self, ys):
        probs_xt_ys, (probs_f, probs_b) = self.hmm.forward_backward(ys)  # P(xt|y1..T) [length, n_states, 1]
        probs_xt_on_ys = probs_xt_ys / np.sum(probs_xt_ys, axis=1, keepdims=True)
        probs_yt_on_xt = self.hmm.mat_o[ys]  # [t, i] = P(y_t+1..T|xt = i) shape = [length, n_states]
        probs_ytp1T_on_xt = probs_b
        probs_xt_y1t_times_ytp1T_on_xtp1 = \
            np.matmul(probs_f[:-1, :, :],    # P(y1..t, x_t)    p(yt+2..T | x_t+1) * P(yt+1 | x_t+1)
                      np.swapaxes(probs_ytp1T_on_xt[1:, :, :] * probs_yt_on_xt[:, :, np.newaxis], 1, 2))
        #  [t, i, j] = P(xt = i, y1..t) * P(yt+1..T|x_t+1 = j)
        probs_xtp1_on_xt = self.hmm.mat_t[np.newaxis, :, :]
        probs_xt_xtp1_ys = probs_xt_y1t_times_ytp1T_on_xtp1 * probs_xtp1_on_xt
        # P(x_t, x_t+1|ys)
        probs_xt_xtp1_on_ys = probs_xt_xtp1_ys / np.sum(probs_xt_xtp1_ys, axis=(1, 2), keepdims=True)
        # probs_xt_xtp1_on_ys = probs_xt_xtp1_ys / np.sum(probs_xt_xtp1_ys, axis=(1, 2))
        real_ys = np.eye(self.hmm.n_obs)[ys]
        real_ys = real_ys.transpose()   # [n_obs, length]
        self.hmm.initial_state = probs_b[0, :, :] / sum(probs_b[0, :, :])
        self.hmm.mat_t = (np.sum(probs_xt_xtp1_on_ys, axis=0) + self.eps) / \
                         (np.sum(probs_xt_on_ys[:-1, :, :], axis=0) + self.hmm.n_states * self.eps)

        # [n_obs, n_states] / [1, n_states]
        self.hmm.mat_o = (np.matmul(real_ys, probs_xt_on_ys[:, :, 0]) + self.eps) / \
                         (np.sum(probs_xt_on_ys, axis=0) + self.hmm.n_obs * self.eps).transpose()
