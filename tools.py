import numpy as np

def log_lin_space(log_start, log_end, log_N, lin_N):
    """ Generate Tuning Space for params
        : param log_start: int
            the begin power for base 10
        : param log_end: int
            the end power for base 10
        : param log_N: int
            numbers to be generated with same power from [10^log_start, 10^log_end]
        : param lin_N: int
            numbers to be generated with same steps and same magnitude, within the logspace

        : return: list
            list of params generated
        ---

        If we with to generate 1 number between [10^-5, 10^-4], and for each
        number generated, find the following 5 numbers with same magnitude, call:

        >>> log_lin_space(-5, -4, 1, 5) 
            array([1.e-05, 2.e-05, 3.e-05, 4.e-05, 5.e-05]) 

        If we with to generate 3 number between [10^-5, 10^-2], and for each
        number generated, find the following 3 numbers with same magnitude, call:

        >>> log_lin_space(-5, -2, 3, 3) 
            array([1.00000000e-05, 2.00000000e-05, 3.00000000e-05, 3.16227766e-04,
                   6.32455532e-04, 9.48683298e-04, 1.00000000e-02, 2.00000000e-02,
                   3.00000000e-02])
    """
    log_space = np.logspace(log_start, log_end, log_N, base=10)
    lin_space = []
    for elem in log_space:
        lin_end = lin_N*elem
        lin_space.append(np.linspace(elem, lin_end, lin_N))

    return np.concatenate(lin_space)