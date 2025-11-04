"""
Kernel functions for Gaussian Process neural population generation.
Simple implementation without GPyTorch complications.
"""

import torch
import numpy as np


def compute_periodic_kernel(x1, x2, lengthscale=0.5, period=2*np.pi):
    """
    Periodic kernel for circular variables.
    
    Parameters:
        x1: [N, 1] tensor
        x2: [M, 1] tensor
        lengthscale: float
        period: float
    
    Returns:
        [N, M] kernel matrix
    """
    # Reshape for broadcasting
    x1 = x1.view(-1, 1)  # [N, 1]
    x2 = x2.view(1, -1)  # [1, M]
    
    # Compute differences
    diff = x1 - x2  # [N, M]
    
    # Periodic distance
    dist = torch.sin(np.pi * diff / period) ** 2
    
    # Apply lengthscale
    K = torch.exp(-2 * dist / (lengthscale ** 2))
    
    return K


def compute_rbf_kernel(x1, x2, lengthscale=2.0):
    """
    RBF kernel for continuous variables.
    
    Parameters:
        x1: [N, 1] tensor
        x2: [M, 1] tensor
        lengthscale: float
    
    Returns:
        [N, M] kernel matrix
    """
    # Reshape for broadcasting
    x1 = x1.view(-1, 1)  # [N, 1]
    x2 = x2.view(1, -1)  # [1, M]
    
    # Compute squared distances
    dist_sq = (x1 - x2) ** 2  # [N, M]
    
    # Apply lengthscale
    K = torch.exp(-dist_sq / (2 * lengthscale ** 2))
    
    return K


def compute_product_kernel(x1, x2, theta_lengthscale=0.5, spatial_lengthscale=2.0):
    """
    Product kernel for mixed selectivity.
    
    Parameters:
        x1: [N, 2] tensor where columns are [theta, location]
        x2: [M, 2] tensor where columns are [theta, location]
        
    Returns:
        [N, M] kernel matrix
    """
    # Split into components
    theta1 = x1[:, 0:1]
    theta2 = x2[:, 0:1]
    spatial1 = x1[:, 1:2]
    spatial2 = x2[:, 1:2]
    
    # Compute individual kernels
    K_theta = compute_periodic_kernel(theta1, theta2, theta_lengthscale)
    K_spatial = compute_rbf_kernel(spatial1, spatial2, spatial_lengthscale)
    
    # Return product
    return K_theta * K_spatial