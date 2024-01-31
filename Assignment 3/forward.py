import numpy as np

# Define the matrices and initial state distribution
transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
observation_matrix_true = np.array([[0.9, 0], [0, 0.2]])
observation_matrix_false = np.array([[0.1, 0], [0, 0.8]])
initial_state_dist = np.array([0.5, 0.5])
evidence = [1, 1, 0, 1, 1]  # In this case, '1' represents seeing the umbrella.


# Perform one step of the forward algorithm
def forward_step(transition_matrix, observation_matrix, prior_state_dist, evidence):

    if len(evidence) == 0:
        return prior_state_dist


    last_step = evidence[-1]
    #set the correct Observation matrix
    if last_step == 1:
        O_t = observation_matrix_true
    else:
        O_t = observation_matrix_false

    T_transposed = np.transpose(transition_matrix)

    prior_state_dist = forward_step(transition_matrix, observation_matrix, prior_state_dist, evidence[:-1])

    non_normalized = O_t@T_transposed@prior_state_dist

    alpha = 1/np.sum(non_normalized)

    return alpha*non_normalized

# Perform the first step with the first piece of evidence
first_step_result = forward_step(transition_matrix, observation_matrix_false, initial_state_dist, evidence)
print(first_step_result)
