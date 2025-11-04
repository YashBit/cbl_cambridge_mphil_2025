"""
Gaussian Process implementation for neural tuning curve generation.
Enhanced version with multiple methods for creating mixed selectivity.
"""

import torch
import numpy as np
from typing import Optional, Literal
from tqdm import tqdm


class NeuralGaussianProcess:
    """
    Generate neural tuning curves using Gaussian Processes with mixed selectivity.
    
    Provides multiple methods:
    - 'direct': Direct construction with guaranteed mixed selectivity
    - 'gp_interaction': GP-based with strong interaction terms
    - 'original': Original implementation (tends toward pure selectivity)
    """
    
    def __init__(
        self,
        n_orientations: int = 20,
        n_locations: int = 4,
        theta_lengthscale: float = 0.3,
        spatial_lengthscale: float = 1.5,
        device: str = 'cpu',
        seed: Optional[int] = None,
        method: Literal['direct', 'gp_interaction', 'original'] = 'direct'
    ):
        self.n_orientations = n_orientations
        self.n_locations = n_locations
        self.theta_lengthscale = theta_lengthscale
        self.spatial_lengthscale = spatial_lengthscale
        self.device = device
        self.method = method
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"Creating input grid: {n_orientations}×{n_locations} = {n_orientations*n_locations} points")
        print(f"Method: {method}")
        self.input_grid = self._create_input_grid()
        
    def _create_input_grid(self) -> torch.Tensor:
        """Create 2D grid of (theta, location) pairs."""
        theta = torch.linspace(-np.pi, np.pi, self.n_orientations)
        locations = torch.arange(self.n_locations).float()
        
        theta_grid, loc_grid = torch.meshgrid(theta, locations, indexing='ij')
        
        grid = torch.stack([
            theta_grid.flatten(),
            loc_grid.flatten()
        ], dim=1)
        
        return grid.to(self.device)
    
    def sample_neurons(self, n_neurons: int = 20) -> np.ndarray:
        """
        Sample neural tuning curves with mixed selectivity.
        
        Dispatches to the appropriate method based on initialization.
        """
        if self.method == 'direct':
            return self._sample_neurons_direct(n_neurons)
        elif self.method == 'gp_interaction':
            return self._sample_neurons_gp_interaction(n_neurons)
        else:  # original
            return self._sample_neurons_original(n_neurons)
    
    def _sample_neurons_direct(self, n_neurons: int = 20) -> np.ndarray:
        """
        Generate neurons with guaranteed mixed selectivity using direct construction.
        
        Core principle: Create tuning that explicitly depends on 
        orientation-location interactions that cannot be separated.
        """
        print(f"\nGenerating {n_neurons} mixed selectivity neurons (direct method)...")
        
        samples = []
        
        for neuron_idx in tqdm(range(n_neurons), desc="Creating neurons"):
            # Initialize tuning curve
            tuning = np.zeros((self.n_orientations, self.n_locations))
            
            # Pattern 1: Rotating preference across locations
            rotation_rate = np.random.uniform(0.5, 2.0)
            base_phase = np.random.uniform(0, 2*np.pi)
            
            for loc in range(self.n_locations):
                for orient_idx in range(self.n_orientations):
                    theta = -np.pi + 2*np.pi * orient_idx / self.n_orientations
                    
                    # Preferred orientation shifts with location
                    pref_theta = base_phase + rotation_rate * loc * np.pi / self.n_locations
                    
                    # Circular distance
                    dist = np.angle(np.exp(1j * (theta - pref_theta)))
                    
                    # Response with location-dependent width
                    width = 0.5 + 0.3 * np.sin(loc * np.pi / 2)
                    response = np.exp(-dist**2 / (2 * width**2))
                    
                    tuning[orient_idx, loc] += response
            
            # Pattern 2: Multiplicative interactions
            n_components = np.random.randint(2, 4)
            
            for _ in range(n_components):
                orient_center = np.random.randint(0, self.n_orientations)
                loc_center = np.random.randint(0, self.n_locations)
                
                for loc in range(self.n_locations):
                    for orient_idx in range(self.n_orientations):
                        orient_dist = min(abs(orient_idx - orient_center),
                                        self.n_orientations - abs(orient_idx - orient_center))
                        loc_dist = abs(loc - loc_center)
                        
                        # Non-separable interaction
                        if (orient_dist + loc_dist) % 2 == 0:
                            interaction = np.exp(-0.3 * (orient_dist * loc_dist))
                        else:
                            interaction = np.exp(-0.5 * (orient_dist + loc_dist)**2)
                        
                        tuning[orient_idx, loc] += np.random.uniform(0.5, 1.5) * interaction
            
            # Pattern 3: XOR-like responses (inherently non-separable)
            if neuron_idx % 2 == 0:
                for loc in range(self.n_locations):
                    for orient_idx in range(self.n_orientations):
                        orient_group = orient_idx < self.n_orientations // 2
                        loc_group = loc < self.n_locations // 2
                        
                        if orient_group != loc_group:  # XOR condition
                            tuning[orient_idx, loc] += np.random.uniform(0.3, 0.7)
            
            # Pattern 4: Sinusoidal interactions with phase coupling
            freq_theta = np.random.uniform(1, 3)
            freq_loc = np.random.uniform(0.5, 1.5)
            phase_coupling = np.random.uniform(0.5, 2.0)
            
            for loc in range(self.n_locations):
                for orient_idx in range(self.n_orientations):
                    theta = -np.pi + 2*np.pi * orient_idx / self.n_orientations
                    
                    response = (1 + np.sin(freq_theta * theta + phase_coupling * loc)) * \
                              (1 + np.cos(freq_loc * loc - phase_coupling * theta))
                    
                    tuning[orient_idx, loc] += 0.2 * response
            
            # Normalize and add noise
            tuning = np.abs(tuning)
            tuning = tuning / (tuning.max() + 1e-8)
            tuning = tuning + np.random.uniform(0.05, 0.15, tuning.shape)
            
            # Apply final nonlinearity
            tuning = np.power(tuning, np.random.uniform(0.7, 1.3))
            
            samples.append(tuning)
        
        return np.array(samples)
    
    def _sample_neurons_gp_interaction(self, n_neurons: int = 20) -> np.ndarray:
        """
        Use GP with strong interaction kernels for mixed selectivity.
        """
        print(f"\nSampling {n_neurons} neurons with interaction GP...")
        
        samples = []
        orientations = torch.linspace(-np.pi, np.pi, self.n_orientations)
        
        for neuron_idx in tqdm(range(n_neurons), desc="GP sampling"):
            tuning = torch.zeros(self.n_orientations, self.n_locations)
            
            # GP with location-varying orientation preference
            for loc in range(self.n_locations):
                lengthscale = 0.3 + 0.5 * np.sin(loc * np.pi / self.n_locations)
                
                K = torch.zeros(self.n_orientations, self.n_orientations)
                for i in range(self.n_orientations):
                    for j in range(self.n_orientations):
                        dist = torch.abs(orientations[i] - orientations[j])
                        dist = torch.min(dist, 2*np.pi - dist)
                        K[i, j] = torch.exp(-dist**2 / (2 * lengthscale**2))
                
                K += 0.01 * torch.eye(self.n_orientations)
                
                L = torch.linalg.cholesky(K)
                z = torch.randn(self.n_orientations)
                sample = L @ z
                
                gain = 1.0 + 0.5 * np.cos(2 * np.pi * loc / self.n_locations + neuron_idx)
                tuning[:, loc] = torch.nn.functional.softplus(sample * gain)
            
            # GP with orientation-varying location preference
            for orient_idx in range(self.n_orientations):
                theta = orientations[orient_idx]
                
                K_loc = torch.zeros(self.n_locations, self.n_locations)
                width = 1.0 + 0.5 * torch.sin(theta)
                
                for i in range(self.n_locations):
                    for j in range(self.n_locations):
                        K_loc[i, j] = torch.exp(-(i-j)**2 / (2 * width**2))
                
                K_loc += 0.01 * torch.eye(self.n_locations)
                
                L_loc = torch.linalg.cholesky(K_loc)
                z_loc = torch.randn(self.n_locations)
                sample_loc = L_loc @ z_loc
                
                tuning[orient_idx, :] *= torch.nn.functional.sigmoid(sample_loc)
            
            # Add conjunction responses
            n_conjunctions = np.random.randint(1, 3)
            for _ in range(n_conjunctions):
                conj_orient = np.random.randint(0, self.n_orientations)
                conj_loc = np.random.randint(0, self.n_locations)
                
                tuning[conj_orient, conj_loc] += torch.abs(torch.randn(1)).item() * 2
                
                for o in range(max(0, conj_orient-2), min(self.n_orientations, conj_orient+3)):
                    for l in range(max(0, conj_loc-1), min(self.n_locations, conj_loc+2)):
                        if o != conj_orient or l != conj_loc:
                            tuning[o, l] *= 0.5
            
            samples.append(tuning.numpy())
        
        return np.array(samples)
    
    def _sample_neurons_original(self, n_neurons: int = 20) -> np.ndarray:
        """
        Original implementation - tends to create more separable tuning.
        Kept for comparison purposes.
        """
        print(f"\nSampling {n_neurons} neurons from GP (original method)...")
        
        from .kernels import compute_product_kernel
        
        with torch.no_grad():
            K = compute_product_kernel(
                self.input_grid, 
                self.input_grid,
                self.theta_lengthscale,
                self.spatial_lengthscale
            )
            
            K = K + 0.1 * torch.eye(K.shape[0]).to(self.device)
            L = torch.linalg.cholesky(K)
            
            samples = []
            
            for i in tqdm(range(n_neurons), desc="Sampling neurons"):
                z = torch.randn(K.shape[0], 1).to(self.device)
                sample = L @ z
                tuning = torch.nn.functional.softplus(sample).reshape(
                    self.n_orientations, self.n_locations
                )
                samples.append(tuning.cpu().numpy())
        
        return np.array(samples)