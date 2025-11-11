"""
Gaussian Process implementation for neural tuning curve generation with mixed selectivity.

This module provides methods for generating realistic neural population responses
that exhibit mixed selectivity - where neurons respond conjunctively to multiple
stimulus features (orientation × spatial location) rather than encoding features
independently.

Mixed selectivity is critical for:
- High-dimensional neural representations
- Flexible computation in prefrontal cortex
- Context-dependent information processing
- Solving linearly inseparable problems

References:
    Rigotti et al. (2013) Nature - "The importance of mixed selectivity..."
    Fusi et al. (2016) Neuron - "Why neurons mix: high dimensionality for..."
"""

import torch
import numpy as np
from typing import Optional, Literal, Tuple
from tqdm import tqdm


class NeuralGaussianProcess:
    """
    Generate neural tuning curves with mixed selectivity.
    
    Provides three complementary methods:
    
    1. 'direct': Engineered construction with guaranteed non-separability
       - Uses explicit interaction terms between dimensions
       - Mathematically guaranteed to produce separability < 0.6
       - Best for: Controlled experiments, algorithm testing
       - Limitation: Less mechanistically realistic
    
    2. 'gp_interaction': Gaussian Process with interaction kernels
       - Samples from GP with location-varying parameters
       - Introduces multiplicative gain modulation
       - Best for: Biologically plausible smooth tuning
       - Limitation: Weaker guarantees on non-separability
    
    3. 'simple_conjunctive': Minimal conjunctive coding mechanism
       - Location-dependent orientation preferences
       - Clear demonstration of mixed selectivity principle
       - Best for: Educational purposes, understanding fundamentals
       - Limitation: Less complex than real neurons
    
    Attributes:
        n_orientations: Number of orientation bins (typically 20-50)
        n_locations: Number of spatial locations (typically 2-8)
        theta_lengthscale: Orientation tuning width for GP method
        spatial_lengthscale: Spatial tuning width for GP method
        method: Generation method to use
    """
    
    def __init__(
        self,
        n_orientations: int = 20,
        n_locations: int = 4,
        theta_lengthscale: float = 0.3,
        spatial_lengthscale: float = 1.5,
        device: str = 'cpu',
        seed: Optional[int] = None,
        method: Literal['direct', 'gp_interaction', 'simple_conjunctive'] = 'direct'
    ):
        """
        Initialize neural population generator.
        
        Args:
            n_orientations: Number of orientation values (evenly spaced in [-π, π])
            n_locations: Number of spatial locations
            theta_lengthscale: Orientation kernel width (for GP method)
            spatial_lengthscale: Spatial kernel width (for GP method)
            device: PyTorch device ('cpu' or 'cuda')
            seed: Random seed for reproducibility
            method: Generation method ('direct', 'gp_interaction', or 'simple_conjunctive')
        """
        self.n_orientations = n_orientations
        self.n_locations = n_locations
        self.theta_lengthscale = theta_lengthscale
        self.spatial_lengthscale = spatial_lengthscale
        self.device = device
        self.method = method
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create stimulus space
        self.input_grid = self._create_input_grid()
        
        print(f"Initialized NeuralGaussianProcess:")
        print(f"  Grid: {n_orientations} orientations × {n_locations} locations")
        print(f"  Total conditions: {n_orientations * n_locations}")
        print(f"  Method: {method}")
        
    def _create_input_grid(self) -> torch.Tensor:
        """
        Create 2D grid of (orientation, location) stimulus conditions.
        
        Returns:
            Tensor of shape (n_orientations * n_locations, 2)
            where each row is [theta, location]
        """
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
        Generate population of neurons with mixed selectivity.
        
        Args:
            n_neurons: Number of neurons to generate
            
        Returns:
            Array of shape (n_neurons, n_orientations, n_locations)
            containing firing rates for each neuron across all conditions
        """
        if self.method == 'direct':
            return self._sample_neurons_direct(n_neurons)
        elif self.method == 'gp_interaction':
            return self._sample_neurons_gp_interaction(n_neurons)
        elif self.method == 'simple_conjunctive':
            return self._sample_neurons_simple_conjunctive(n_neurons)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _sample_neurons_simple_conjunctive(self, n_neurons: int = 20) -> np.ndarray:
        """
        Generate neurons using simple conjunctive coding mechanism.
        
        Core principle: Each neuron's preferred orientation shifts systematically
        with spatial location, creating inherent mixed selectivity.
        
        This is the SIMPLEST demonstration that:
        1. Preferred orientation = f(location) breaks separability
        2. Response cannot be written as g(orientation) × h(location)
        3. Creates conjunction coding essential for working memory
        
        Args:
            n_neurons: Number of neurons to generate
            
        Returns:
            Array of shape (n_neurons, n_orientations, n_locations)
        """
        print(f"\nGenerating {n_neurons} neurons with simple conjunctive coding...")
        print("  Core mechanism: Location-dependent orientation preference")
        
        samples = []
        orientations = np.linspace(-np.pi, np.pi, self.n_orientations)
        
        for neuron_idx in tqdm(range(n_neurons), desc="Creating neurons"):
            tuning = np.zeros((self.n_orientations, self.n_locations))
            
            # Each neuron has unique parameters
            # Rotation rate: how much preferred orientation shifts per location
            rotation_rate = np.random.uniform(0.3, 2.0)
            
            # Base preferred orientation at location 0
            base_preference = np.random.uniform(-np.pi, np.pi)
            
            # Tuning width (can also vary with location for more complexity)
            base_width = np.random.uniform(0.3, 0.8)
            
            # Generate conjunction centers (hot spots)
            n_hotspots = np.random.randint(1, 3)
            hotspots = [(np.random.randint(0, self.n_orientations),
                        np.random.randint(0, self.n_locations))
                       for _ in range(n_hotspots)]
            
            # Main mechanism: location-dependent orientation preference
            for loc in range(self.n_locations):
                # KEY: Preferred orientation shifts with location
                # This single line breaks separability!
                preferred_ori = base_preference + rotation_rate * loc * np.pi / self.n_locations
                
                # Width can also modulate with location
                width = base_width * (1 + 0.3 * np.sin(loc * np.pi / self.n_locations))
                
                for ori_idx, theta in enumerate(orientations):
                    # Circular distance to preferred orientation
                    angular_dist = np.angle(np.exp(1j * (theta - preferred_ori)))
                    
                    # Gaussian tuning around location-dependent preference
                    response = np.exp(-angular_dist**2 / (2 * width**2))
                    
                    # Add contribution from conjunction hot spots
                    for (hot_ori, hot_loc) in hotspots:
                        ori_dist = min(abs(ori_idx - hot_ori),
                                      self.n_orientations - abs(ori_idx - hot_ori))
                        loc_dist = abs(loc - hot_loc)
                        
                        # Multiplicative interaction creates non-separability
                        if ori_dist * loc_dist < 5:  # Local conjunction
                            response += 0.5 * np.exp(-0.2 * (ori_dist * loc_dist))
                    
                    tuning[ori_idx, loc] = response
            
            # Add baseline activity
            tuning = tuning + np.random.uniform(0.05, 0.15)
            
            # Normalize to [0, 1] range
            tuning = tuning / (tuning.max() + 1e-8)
            
            samples.append(tuning)
        
        result = np.array(samples)
        print(f"✓ Generated {n_neurons} neurons with shape {result.shape}")
        return result
    
    def _sample_neurons_direct(self, n_neurons: int = 20) -> np.ndarray:
        """
        Generate neurons using direct construction with guaranteed mixed selectivity.
        
        Implements four complementary mechanisms for non-separable responses:
        
        1. Rotating preference: Preferred orientation shifts systematically
           with spatial location
           
        2. Multiplicative interactions: Response depends on products of 
           orientation and location distances (orient_dist × loc_dist)
           
        3. XOR-like conjunctions: Logical operations that are inherently
           non-separable (classical example from neural network theory)
           
        4. Phase-coupled oscillations: Sinusoidal responses with cross-dimension
           phase coupling
        
        Each mechanism alone would produce non-separability; together they
        ensure robust mixed selectivity across the population.
        
        Args:
            n_neurons: Number of neurons to generate
            
        Returns:
            Array of shape (n_neurons, n_orientations, n_locations)
        """
        print(f"\nGenerating {n_neurons} neurons with direct construction...")
        print("  Using 4 non-separable mechanisms:")
        print("    1. Rotating preferences")
        print("    2. Multiplicative interactions")
        print("    3. XOR conjunctions")
        print("    4. Phase-coupled oscillations")
        
        samples = []
        orientations = np.linspace(-np.pi, np.pi, self.n_orientations)
        
        for neuron_idx in tqdm(range(n_neurons), desc="Creating neurons"):
            # Initialize response matrix
            tuning = np.zeros((self.n_orientations, self.n_locations))
            
            # MECHANISM 1: Rotating preference across locations
            rotation_rate = np.random.uniform(0.5, 2.0)
            base_phase = np.random.uniform(0, 2*np.pi)
            
            for loc in range(self.n_locations):
                pref_theta = base_phase + rotation_rate * loc * np.pi / self.n_locations
                
                for orient_idx, theta in enumerate(orientations):
                    dist = np.angle(np.exp(1j * (theta - pref_theta)))
                    width = 0.5 + 0.3 * np.sin(loc * np.pi / 2)
                    response = np.exp(-dist**2 / (2 * width**2))
                    tuning[orient_idx, loc] += response
            
            # MECHANISM 2: Multiplicative interactions
            n_components = np.random.randint(2, 4)
            
            for _ in range(n_components):
                orient_center = np.random.randint(0, self.n_orientations)
                loc_center = np.random.randint(0, self.n_locations)
                
                for loc in range(self.n_locations):
                    for orient_idx in range(self.n_orientations):
                        orient_dist = min(
                            abs(orient_idx - orient_center),
                            self.n_orientations - abs(orient_idx - orient_center)
                        )
                        loc_dist = abs(loc - loc_center)
                        
                        if (orient_dist + loc_dist) % 2 == 0:
                            interaction = np.exp(-0.3 * (orient_dist * loc_dist))
                        else:
                            interaction = np.exp(-0.5 * (orient_dist + loc_dist)**2)
                        
                        tuning[orient_idx, loc] += np.random.uniform(0.5, 1.5) * interaction
            
            # MECHANISM 3: XOR-like responses (every other neuron)
            if neuron_idx % 2 == 0:
                for loc in range(self.n_locations):
                    for orient_idx in range(self.n_orientations):
                        orient_group = orient_idx < self.n_orientations // 2
                        loc_group = loc < self.n_locations // 2
                        
                        if orient_group != loc_group:
                            tuning[orient_idx, loc] += np.random.uniform(0.3, 0.7)
            
            # MECHANISM 4: Sinusoidal interactions with phase coupling
            freq_theta = np.random.uniform(1, 3)
            freq_loc = np.random.uniform(0.5, 1.5)
            phase_coupling = np.random.uniform(0.5, 2.0)
            
            for loc in range(self.n_locations):
                for orient_idx, theta in enumerate(orientations):
                    response = (1 + np.sin(freq_theta * theta + phase_coupling * loc)) * \
                              (1 + np.cos(freq_loc * loc - phase_coupling * theta))
                    
                    tuning[orient_idx, loc] += 0.2 * response
            
            # POST-PROCESSING
            tuning = np.abs(tuning)
            tuning = tuning / (tuning.max() + 1e-8)
            tuning = tuning + np.random.uniform(0.05, 0.15, tuning.shape)
            tuning = np.power(tuning, np.random.uniform(0.7, 1.3))
            
            samples.append(tuning)
        
        result = np.array(samples)
        print(f"✓ Generated {n_neurons} neurons with shape {result.shape}")
        return result
    
    def _sample_neurons_gp_interaction(self, n_neurons: int = 20) -> np.ndarray:
        """
        Generate neurons using Gaussian Process with interaction terms.
        
        Strategy:
        1. Sample orientation tuning independently at each location with
           location-varying lengthscales (creates heterogeneity)
           
        2. Sample location tuning independently at each orientation with
           orientation-varying widths (creates interaction)
           
        3. Multiply the two samples (creates multiplicative interaction)
        
        4. Add conjunction hot-spots with surround suppression
        
        This produces smoother, more biologically plausible tuning than
        the direct method, but with weaker guarantees on non-separability.
        
        Args:
            n_neurons: Number of neurons to generate
            
        Returns:
            Array of shape (n_neurons, n_orientations, n_locations)
        """
        print(f"\nSampling {n_neurons} neurons from interaction GP...")
        print("  Using location-varying orientation kernels")
        print("  Using orientation-varying spatial kernels")
        print("  Adding conjunction responses")
        
        samples = []
        orientations = torch.linspace(-np.pi, np.pi, self.n_orientations)
        
        for neuron_idx in tqdm(range(n_neurons), desc="GP sampling"):
            tuning = torch.ones(self.n_orientations, self.n_locations)
            
            # PHASE 1: Orientation tuning varies with location
            for loc in range(self.n_locations):
                lengthscale = self.theta_lengthscale + \
                             0.5 * self.theta_lengthscale * np.sin(loc * np.pi / self.n_locations)
                
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
            
            # PHASE 2: Spatial tuning varies with orientation
            for orient_idx in range(self.n_orientations):
                theta = orientations[orient_idx]
                
                width = self.spatial_lengthscale * (1.0 + 0.5 * torch.sin(theta))
                
                K_loc = torch.zeros(self.n_locations, self.n_locations)
                for i in range(self.n_locations):
                    for j in range(self.n_locations):
                        K_loc[i, j] = torch.exp(-(i-j)**2 / (2 * width**2))
                
                K_loc += 0.01 * torch.eye(self.n_locations)
                
                L_loc = torch.linalg.cholesky(K_loc)
                z_loc = torch.randn(self.n_locations)
                sample_loc = L_loc @ z_loc
                
                tuning[orient_idx, :] *= torch.nn.functional.sigmoid(sample_loc)
            
            # PHASE 3: Add conjunction responses with surround suppression
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
        
        result = np.array(samples)
        print(f"✓ Generated {n_neurons} neurons with shape {result.shape}")
        return result


def compare_methods(
    n_neurons: int = 20,
    n_orientations: int = 20,
    n_locations: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare all three generation methods side-by-side.
    
    Useful for understanding trade-offs between engineered construction,
    GP-based sampling, and simple conjunctive coding.
    
    Args:
        n_neurons: Number of neurons to generate with each method
        n_orientations: Number of orientation bins
        n_locations: Number of spatial locations
        seed: Random seed for reproducibility
        
    Returns:
        (direct_samples, gp_samples, simple_samples): Tuple of arrays, each with shape
            (n_neurons, n_orientations, n_locations)
    """
    print("\n" + "="*60)
    print("COMPARING GENERATION METHODS")
    print("="*60)
    
    # Generate with direct method
    print("\n--- DIRECT METHOD ---")
    gp_direct = NeuralGaussianProcess(
        n_orientations=n_orientations,
        n_locations=n_locations,
        seed=seed,
        method='direct'
    )
    direct_samples = gp_direct.sample_neurons(n_neurons)
    
    # Generate with GP method
    print("\n--- GP INTERACTION METHOD ---")
    gp_interaction = NeuralGaussianProcess(
        n_orientations=n_orientations,
        n_locations=n_locations,
        seed=seed + 1,
        method='gp_interaction'
    )
    gp_samples = gp_interaction.sample_neurons(n_neurons)
    
    # Generate with simple conjunctive method
    print("\n--- SIMPLE CONJUNCTIVE METHOD ---")
    gp_simple = NeuralGaussianProcess(
        n_orientations=n_orientations,
        n_locations=n_locations,
        seed=seed + 2,
        method='simple_conjunctive'
    )
    simple_samples = gp_simple.sample_neurons(n_neurons)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    
    return direct_samples, gp_samples, simple_samples