import numpy as np


def greadient_base_stability(reward_lst):
    return np.std(np.diff(reward_lst, n=2) / 2)  # range [0, +inf]


def find_first_window_starting_point(reward_lst, window_size):
    for i in range(len(reward_lst) - window_size - 1):
        window = reward_lst[i:i+window_size]
        window_var = np.var(window)

        # Check if all values in the window are greater than 0
        if all(value > 0 for value in window) and window_var < 50:
            return i  # Return the starting point of the first satisfying window

    return None  # Return None if no satisfying window is found


def convergence_eval(reward_lst, window_size):
    idx = find_first_window_starting_point(reward_lst, window_size)
    
    is_convergent = True
    
    if idx is None:
        idx = len(reward_lst) - 1 - window_size
        is_convergent = False
    
    # convergence ratio represents the speed of convergence
    convergence_ratio = idx / (len(reward_lst) - 1 - window_size)
    
    # prevent the drop after the convergence
    rest_var = np.var(reward_lst[idx:])
    # naively clip the range to 1000
    rest_var = 1000.0 if rest_var > 1000.0 else rest_var
    rest_var_ratio = rest_var / 1000                    # range [0, 1]
    
    # get the convergence mean reward
    window_mean = np.mean(reward_lst[idx:idx+window_size])
    window_mean_ratio = abs(300 - window_mean) / 300     # range [0, 1]
    
    return rest_var_ratio + window_mean_ratio


def sc_score(reward_lst, window_size=600):
    # smooth metric range: [0, +inf]
    # convergence_eval range: [0, 2]
    return greadient_base_stability(reward_lst) + convergence_eval(reward_lst, window_size)