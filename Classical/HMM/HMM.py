import numpy as np
class HMM:
    def __init__(self, n_states, n_obs,
                 initial_state: np.ndarray = None,
                 mat_t: np.ndarray = None,
                 mat_o: np.ndarray = None):
        self.n_states = n_states
        self.n_obs = n_obs
        if mat_t is None:
            self.mat_t = np.random.uniform(0, 1, [n_states, n_states])
            self.mat_t /= np.sum(self.mat_t, axis=0, keepdims=True)  # [1, n_states]
            # Transition matrix's column sum must be 1
        else:
            assert mat_t.shape == (n_states, n_states), "Transition matrix shape wrong"
        if mat_o is None:
            self.mat_o = np.random.uniform(0, 1, [n_obs, n_states])
        else:
            assert mat_o.shape == (n_obs, n_states), "Emission matrix shape wrong"
        if initial_state is None:
            self.intial_state = np.random.uniform(0, 1, [n_states, 1])
        else:
            assert initial_state.shape == (n_states, 1), "Initial state matrix shape wrong"
