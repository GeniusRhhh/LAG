import logging
import os
import numpy as np
from typing import Dict, List
from abc import ABC, abstractstaticmethod
from scipy.optimize import linprog

def get_algorithm(algo_name):
    if algo_name == 'sp':
        return SP
    elif algo_name == 'fsp':
        return FSP
    elif algo_name == 'pfsp':
        return PFSP
    elif algo_name == 'psro':
        return PSRO
    else:
        raise NotImplementedError("Unknown algorithm {}".format(algo_name))

class SelfplayAlgorithm(ABC):
    @abstractstaticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        pass

    @abstractstaticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass

class SP(SelfplayAlgorithm):
    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return list(agents_elo.keys())[-1]

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass

class FSP(SelfplayAlgorithm):
    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return np.random.choice(list(agents_elo.keys()))

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass

class PFSP(SelfplayAlgorithm):
    @staticmethod
    def choose(agents_elo: Dict[str, float], lam=1, s=100, **kwargs) -> str:
        if not agents_elo:
            logging.warning("Agents_elo is empty in PFSP.choose; returning default '0'.")
            return '0'
        history_elo = np.array(list(agents_elo.values()))
        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
        k = float(len(sample_probs) + 1)
        meta_solver_probs = np.exp(lam / k * sample_probs) / np.sum(np.exp(lam / k * sample_probs))
        logging.debug(f"PFSP choose probabilities: {dict(zip(agents_elo.keys(), meta_solver_probs))}")
        opponent_idx = np.random.choice(a=list(agents_elo.keys()), size=1, p=meta_solver_probs).item()
        return opponent_idx

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]]) -> None:
        pass

class PSRO(SelfplayAlgorithm):
    def __init__(self, policy_pool: Dict[str, float], payoff_matrix: np.ndarray = None):
        self.policy_pool = policy_pool
        n = max(1, len(policy_pool))
        self.payoff_matrix = payoff_matrix if payoff_matrix is not None else np.full((n, n), 0.5)

    def infer_payoff_matrix(self, keys: List[str]) -> np.ndarray:
        n = len(keys)
        inferred_matrix = self.payoff_matrix.copy()
        for i in range(n):
            for j in range(n):
                if inferred_matrix[i, j] == 0.5 and i != j:
                    for k in range(n):
                        if inferred_matrix[i, k] != 0.5 and inferred_matrix[k, j] != 0.5:
                            inferred = min(inferred_matrix[i, k], inferred_matrix[k, j])
                            inferred_matrix[i, j] = inferred
                            inferred_matrix[j, i] = 1 - inferred
                            logging.debug(f"Inferred payoff [{i}, {j}] = {inferred:.2f}, "
                                         f"[{j}, {i}] = {1 - inferred:.2f} via path {i}->{k}->{j}")
                            break
        return inferred_matrix

    def choose(self, agents_elo: Dict[str, float]) -> str:
        if not agents_elo:
            logging.warning("Policy pool is empty; returning '0'.")
            return '0'
        n = len(agents_elo)
        if self.payoff_matrix.shape != (n, n):
            logging.info(f"Resizing payoff matrix to ({n}, {n})")
            new_matrix = np.full((n, n), 0.5)
            old_n = min(self.payoff_matrix.shape[0], n)
            new_matrix[:old_n, :old_n] = self.payoff_matrix[:old_n, :old_n]
            self.payoff_matrix = new_matrix

        if np.all(self.payoff_matrix == 0.5):
            logging.warning("Payoff matrix is unchanged; choosing randomly.")
            return np.random.choice(list(agents_elo.keys()))

        epsilon = 1e-3  # 增加扰动以提高探索性
        perturbed_matrix = self.payoff_matrix + epsilon * np.random.rand(n, n)
        c = np.array([0] * n + [-1])
        A_ub = np.hstack((-perturbed_matrix.T, np.ones((n, 1))))
        b_ub = np.zeros(n)
        A_eq = np.array([[1] * n + [0]])
        b_eq = np.array([1])
        bounds = [(0, 1)] * n + [(None, None)]

        logging.info(f"linprog inputs: c={c.shape}, A_ub={A_ub.shape}, b_ub={b_ub.shape}")
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')
        if res.success:
            strategy = res.x[:-1]
            if strategy.sum() == 0 or np.any(np.isnan(strategy)):
                logging.warning("Invalid strategy probabilities; choosing randomly.")
                return np.random.choice(list(agents_elo.keys()))
            strategy = strategy / strategy.sum()
            logging.info(f"PSRO choose probabilities: {dict(zip(agents_elo.keys(), strategy))}")
            return np.random.choice(list(agents_elo.keys()), p=strategy)
        logging.warning(f"linprog failed: {res.message}; choosing randomly.")
        return np.random.choice(list(agents_elo.keys()))

    def update(self, agents_elo: Dict[str, float], eval_results: Dict) -> None:
        n = len(agents_elo)
        if n == 0:
            logging.warning("Policy pool is empty, cannot update payoff matrix.")
            self.payoff_matrix = np.full((1, 1), 0.5)
            return

        if self.payoff_matrix.shape != (n, n):
            logging.info(f"Resizing payoff matrix from {self.payoff_matrix.shape} to ({n}, {n})")
            new_matrix = np.full((n, n), 0.5)
            old_n = min(self.payoff_matrix.shape[0], n)
            new_matrix[:old_n, :old_n] = self.payoff_matrix[:old_n, :old_n]
            self.payoff_matrix = new_matrix

        keys = list(agents_elo.keys())
        latest_idx = n - 1
        logging.debug(f"Before update: payoff_matrix =\n{self.payoff_matrix}")

        for key, results in eval_results.items():
            if isinstance(key, tuple):
                opp1, opp2 = key
                if opp1 not in keys or opp2 not in keys:
                    logging.warning(f"Opponent pair {key} not in policy pool; skipping.")
                    continue
                idx1, idx2 = keys.index(opp1), keys.index(opp2)
                result = np.mean(results)
                self.payoff_matrix[idx1, idx2] = result
                self.payoff_matrix[idx2, idx1] = 1 - result
                logging.debug(f"Updated [{idx1}, {idx2}] = {result:.2f}, [{idx2}, {idx1}] = {1 - result:.2f}")
            else:
                if key not in keys:
                    logging.warning(f"Opponent {key} not in policy pool; skipping.")
                    continue
                opponent_idx = keys.index(key)
                result = np.mean(results)
                self.payoff_matrix[latest_idx, opponent_idx] = result
                self.payoff_matrix[opponent_idx, latest_idx] = 1 - result
                logging.debug(
                    f"Updated [{latest_idx}, {opponent_idx}] = {result:.2f}, "
                    f"[{opponent_idx}, {latest_idx}] = {1 - result:.2f}"
                )

        self.payoff_matrix[latest_idx, latest_idx] = 0.5
        np.fill_diagonal(self.payoff_matrix, 0.5)

        self.payoff_matrix = self.infer_payoff_matrix(keys)

        unevaluated_ratio = np.sum(self.payoff_matrix == 0.5) / self.payoff_matrix.size
        logging.info(f"Payoff matrix unevaluated ratio: {unevaluated_ratio:.2f}")
        logging.info(f"Updated payoff matrix:\n{self.payoff_matrix}")