import numpy as np
from python.utils import generate_random_permutation, generate_random_rigid_transformation, transform, permute, add_noise
from python.view_data import plot_comparison

if __name__ == '__main__':
    NOISE_SCALE = 0.05

    X = np.loadtxt(open("../assets/cat.csv", "rb"), delimiter=",")

    N = X.shape[0]
    d = X.shape[1]

    # Generate random rigid transformation, permutation, and add noise

    _L, _t = generate_random_rigid_transformation(d=d)
    _P = generate_random_permutation(N=N)

    Y = transform(X, _L, _t)

    Y = add_noise(Y, sigma=NOISE_SCALE)

    Y = permute(Y, _P)

    # Generate random "solution"

    L, t = generate_random_rigid_transformation(d=d)
    P = generate_random_permutation(N=N)
    
    Z = permute(transform(X, L, t), P)
    
    metric = np.linalg.norm(Y - Z)
    
    print(f"Metric: {metric}")
    
    # View the data

    
    plot_comparison(X, Y, Z)
    
    
    
    
