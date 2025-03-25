import numpy as np
import pandas as pd
import torch


def generate_piecewise_function(
    true_x: torch.Tensor,
    n_lengthscales: int,
    wavelength_min_m: float,
    wavelength_max_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a piecewise function composed of different sine waves with varying wavelengths.
    Each piece has a different wavelength and is stitched together continuously.

    Args:
        true_x: Tensor of x coordinates
        n_lengthscales: Number of different wavelengths/pieces to use
        wavelength_min_m: Minimum wavelength
        wavelength_max_m: Maximum wavelength

    Returns:
        true_y: The resulting piecewise function values
        input_sines_y: Individual sine waves used to construct the function

    """
    # Generate wavelengths logarithmically spaced
    input_wavelengths = torch.logspace(
        np.log10(wavelength_min_m), np.log10(wavelength_max_m), n_lengthscales
    )

    # Create base sine waves for each wavelength
    input_sines_y = torch.stack([torch.sin(true_x * w) for w in input_wavelengths])

    # Create boundaries for each piece
    chunk_boundaries = torch.linspace(true_x.min(), true_x.max(), n_lengthscales + 1)

    # Randomly assign wavelengths to chunks
    wavelength_assignment = torch.randperm(n_lengthscales)

    # Initialize matrices for constructing piecewise function
    selection_matrix = torch.zeros((n_lengthscales, len(true_x)))
    offset_adjustments = torch.zeros_like(true_x)

    # Construct piecewise function
    for i in range(n_lengthscales):
        # Define chunk boundaries
        chunk_mask = (true_x >= chunk_boundaries[i]) & (
            true_x <= chunk_boundaries[i + 1]
        )
        assigned_wave = wavelength_assignment[i]

        # Select which sine wave to use for this chunk
        selection_matrix[assigned_wave, chunk_mask] = 1

        # Calculate offset to ensure continuity between chunks
        if i > 0:
            boundary_idx = torch.argmin(torch.abs(true_x - chunk_boundaries[i]))
            current_value = input_sines_y[assigned_wave, boundary_idx]
            prev_value = input_sines_y[wavelength_assignment[i - 1], boundary_idx - 1]
            offset = current_value - prev_value
            offset_adjustments[true_x >= chunk_boundaries[i]] += offset

    # Combine everything to get final function
    true_y = torch.sum(selection_matrix * input_sines_y, dim=0) - offset_adjustments

    return true_y, input_sines_y


def sample_noisy_data(
    true_x: torch.Tensor,
    true_y: torch.Tensor,
    n_samples: int,
    noise_amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points from the true function and add Gaussian noise.

    Args:
        true_x: x coordinates of true function
        true_y: y values of true function
        n_samples: Number of points to sample
        noise_amplitude: Standard deviation of Gaussian noise

    Returns:
        train_x: Sampled x coordinates
        train_y: Sampled y values with noise

    """
    # Sample indices and sort them for better visualization
    idx_sample = torch.randint(high=len(true_x), size=(n_samples,))
    idx_sample = sorted(idx_sample)

    # Get training points
    train_x = true_x[idx_sample]
    train_y = true_y[idx_sample]

    # Add Gaussian noise
    noise = noise_amplitude * torch.randn_like(train_x)
    train_y = train_y + noise

    return train_x, train_y


def load_true_and_obs_tensors(
    filename_true: str, filename_obs: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df_true = pd.read_csv(filename_true)
    df_obs = pd.read_csv(filename_obs)

    # Convert to tensors
    true_x = torch.tensor(df_true["x"].values, dtype=torch.float32)
    true_y = torch.tensor(df_true["y"].values, dtype=torch.float32)
    obs_x = torch.tensor(df_obs["x"].values, dtype=torch.float32)
    obs_y = torch.tensor(df_obs["y"].values, dtype=torch.float32)

    return true_x, true_y, obs_x, obs_y
