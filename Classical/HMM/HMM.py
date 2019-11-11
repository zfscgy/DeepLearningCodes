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
        :param mat_t:  [n_states, n_states] mat_t[i,j] = P(X(t-1) = i|X(t) = j)
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
        :return: [P(x1,y1), P(x2,y1..y2) ... P(xn, y1..yn)]
        """
        probs = [self.initial_state]
        for i in range(len(ys)):
            prob = self.mat_o[ys[i], :].reshape([self.n_states, 1]) * \
                np.array(np.matmul(self.mat_t.transpose(), probs[i]))
            probs.append(prob)
        return probs[1:]

    def backward(self, ys):
        """

        :param ys:
        :return: [P(y1...yn|x0), P(y2...yn|x1) ,... P(yn|xn-1)]
        """
        probs = [np.ones([self.n_states, 1])]
        for i in reversed(range(len(ys))):
            prob = np.matmul(self.mat_t, self.mat_o[ys[i], :].reshape([self.n_states, 1]) * probs[0])
            probs.insert(0, prob)
        return probs[:-1]

    def forward_backward(self, ys):
        probs_x_ys = self.forward(ys)
        probs_ys_on_x = self.backward(ys)
        probs = []
        for i in range(len(ys)):
            # P(xi, y1..i) * P(yi+1...n|xi)
            prob = probs_x_ys[i] * probs_ys_on_x[i]
            probs.append(prob/sum(prob))
        return probs
    