import numpy as np

# Define the matrices and initial state distribution
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
observation_matrices = {
    1: np.array([[0.9, 0], [0, 0.2]]),  # True observation matrix
    0: np.array([[0.1, 0], [0, 0.8]])  # False observation matrix
}
initial_state_dist = np.array([0.5, 0.5])
# In this case, '1' represents seeing the umbrella.
evidence = [1, 1, 0, 1, 1]


# Perform one step of the forward algorithm
def forward_step(transition_matrix, observation_matrices, prior_state_dist, evidence):
    # If there is no evidence, return the prior state distribution
    if len(evidence) == 0:
        return prior_state_dist

    last_step = evidence[-1]
    # Access the correct observation matrix based on the last evidence
    O_t = observation_matrices[last_step]

    T_transposed = np.transpose(transition_matrix)
    # Recursively call the forward algorithm with the last piece of evidence removed
    forward_step_1_through_t = forward_step(transition_matrix, observation_matrices, prior_state_dist, evidence[:-1])

    # Calculate the non-normalized forward message given the formula from 14.12
    non_normalized = O_t @ T_transposed @ forward_step_1_through_t
    alpha = 1 / np.sum(non_normalized)

    return alpha * non_normalized


# Perform the first step with the first piece of evidence
first_step_result = forward_step(transition_matrix, observation_matrices, initial_state_dist, evidence)
print(first_step_result)
