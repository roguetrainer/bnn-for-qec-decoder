"""
Bayesian Neural Network for Quantum Error Correction Decoding

This implementation uses Bayesian deep learning to decode quantum error syndromes
with uncertainty quantification, which is crucial for adaptive QEC protocols.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """Configuration for the BNN QEC decoder"""
    syndrome_size: int
    hidden_dims: List[int]
    num_samples: int = 50  # Number of forward passes for uncertainty estimation
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    learning_rate: float = 1e-3
    kl_weight: float = 1.0 / 1000  # Weight for KL divergence term


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer implementing variational inference.
    
    Uses the local reparameterization trick for efficient sampling.
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log std)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_std = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_std = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters with small random values"""
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_log_std, -5)  # Small initial std
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_std, -5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with reparameterization trick.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        if self.training:
            # Sample weights and biases during training
            weight_std = torch.exp(self.weight_log_std)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.exp(self.bias_log_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean values during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        KL(q(w) || p(w)) for Gaussian distributions
        """
        weight_std = torch.exp(self.weight_log_std)
        bias_std = torch.exp(self.bias_log_std)
        
        # KL for weights
        kl_weight = torch.sum(
            torch.log(self.prior_std / weight_std) +
            (weight_std**2 + self.weight_mu**2) / (2 * self.prior_std**2) - 0.5
        )
        
        # KL for biases
        kl_bias = torch.sum(
            torch.log(self.prior_std / bias_std) +
            (bias_std**2 + self.bias_mu**2) / (2 * self.prior_std**2) - 0.5
        )
        
        return kl_weight + kl_bias


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network for QEC decoding.
    
    Predicts error corrections with uncertainty estimates.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # Build Bayesian layers
        layers = []
        input_dim = config.syndrome_size
        
        for hidden_dim in config.hidden_dims:
            layers.append(BayesianLinear(input_dim, hidden_dim, config.prior_std))
            input_dim = hidden_dim
        
        # Output layer (number of qubits for correction)
        # For simplicity, we predict corrections for the same number of qubits as syndrome bits
        layers.append(BayesianLinear(input_dim, config.syndrome_size, config.prior_std))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            syndrome: Syndrome measurements (batch_size, syndrome_size)
            
        Returns:
            Correction predictions (batch_size, syndrome_size)
        """
        x = syndrome
        
        # Pass through all layers except the last with ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Final layer without activation (logits)
        x = self.layers[-1](x)
        
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence across all layers"""
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def predict_with_uncertainty(self, syndrome: torch.Tensor, 
                                num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            syndrome: Syndrome measurements (batch_size, syndrome_size)
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            mean_prediction: Mean correction prediction
            uncertainty: Standard deviation (epistemic uncertainty)
        """
        if num_samples is None:
            num_samples = self.config.num_samples
            
        self.eval()
        predictions = []
        
        # Enable dropout-like behavior by setting training mode temporarily
        for _ in range(num_samples):
            self.train()  # Enable stochastic forward passes
            with torch.no_grad():
                pred = torch.sigmoid(self(syndrome))
            predictions.append(pred)
        
        self.eval()
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_prediction, uncertainty


class BNNQECDecoder:
    """
    Complete Bayesian Neural Network QEC Decoder system.
    
    Handles training, decoding, and uncertainty-aware error correction.
    """
    
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.model = BayesianNeuralNetwork(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.training_history = {'loss': [], 'kl': [], 'nll': []}
        
    def elbo_loss(self, syndrome: torch.Tensor, target_correction: torch.Tensor, 
                  num_batches: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Evidence Lower Bound (ELBO) loss.
        
        ELBO = E[log p(y|x,w)] - KL(q(w)||p(w))
        
        Args:
            syndrome: Input syndromes
            target_correction: Target corrections
            num_batches: Number of batches (for KL weight scaling)
            
        Returns:
            total_loss, nll_loss, kl_loss
        """
        # Forward pass
        logits = self.model(syndrome)
        
        # Negative log-likelihood (binary cross-entropy)
        nll_loss = F.binary_cross_entropy_with_logits(logits, target_correction)
        
        # KL divergence (scaled by number of batches)
        kl_loss = self.model.kl_divergence() / num_batches
        
        # Total ELBO loss
        total_loss = nll_loss + self.config.kl_weight * kl_loss
        
        return total_loss, nll_loss, kl_loss
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> dict:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_nll = 0.0
        num_batches = len(train_loader)
        
        for syndrome, correction in train_loader:
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, nll, kl = self.elbo_loss(syndrome, correction, num_batches)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_kl += kl.item()
            epoch_nll += nll.item()
        
        # Average over batches
        metrics = {
            'loss': epoch_loss / num_batches,
            'kl': epoch_kl / num_batches,
            'nll': epoch_nll / num_batches
        }
        
        return metrics
    
    def train(self, train_loader: torch.utils.data.DataLoader, num_epochs: int, 
              verbose: bool = True):
        """
        Train the BNN decoder.
        
        Args:
            train_loader: DataLoader with (syndrome, correction) pairs
            num_epochs: Number of training epochs
            verbose: Whether to print progress
        """
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader)
            
            # Store history
            self.training_history['loss'].append(metrics['loss'])
            self.training_history['kl'].append(metrics['kl'])
            self.training_history['nll'].append(metrics['nll'])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"NLL={metrics['nll']:.4f}, "
                      f"KL={metrics['kl']:.4f}")
    
    def decode(self, syndrome: np.ndarray, 
               return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Decode a syndrome measurement to error correction.
        
        Args:
            syndrome: Syndrome measurement array
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            correction: Predicted error correction
            uncertainty: Uncertainty estimates (if requested)
        """
        # Convert to tensor
        syndrome_tensor = torch.tensor(syndrome, dtype=torch.float32)
        if syndrome_tensor.dim() == 1:
            syndrome_tensor = syndrome_tensor.unsqueeze(0)
        
        # Get prediction with uncertainty
        mean_pred, uncertainty = self.model.predict_with_uncertainty(syndrome_tensor)
        
        # Threshold at 0.5 for binary correction
        correction = (mean_pred > 0.5).cpu().numpy().astype(int)
        
        if return_uncertainty:
            return correction, uncertainty.cpu().numpy()
        else:
            return correction, None
    
    def adaptive_decode(self, syndrome: np.ndarray, 
                       uncertainty_threshold: float = 0.3) -> Tuple[np.ndarray, bool]:
        """
        Adaptive decoding that flags high-uncertainty cases for additional processing.
        
        Args:
            syndrome: Syndrome measurement
            uncertainty_threshold: Threshold for flagging uncertain predictions
            
        Returns:
            correction: Predicted correction
            needs_review: Whether prediction has high uncertainty
        """
        correction, uncertainty = self.decode(syndrome, return_uncertainty=True)
        
        # Check if any qubit has high uncertainty
        max_uncertainty = np.max(uncertainty)
        needs_review = max_uncertainty > uncertainty_threshold
        
        return correction, needs_review
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.training_history['loss'])
        axes[0].set_title('Total Loss (ELBO)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        axes[1].plot(self.training_history['nll'])
        axes[1].set_title('Negative Log-Likelihood')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('NLL')
        axes[1].grid(True)
        
        axes[2].plot(self.training_history['kl'])
        axes[2].set_title('KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL')
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig


# =============================================================================
# Synthetic Data Generation for Surface Code QEC
# =============================================================================

class SurfaceCodeDataGenerator:
    """
    Generate synthetic training data for surface code QEC.
    
    Simulates syndrome measurements and their corresponding corrections.
    """
    
    def __init__(self, code_distance: int, error_rate: float = 0.01):
        self.code_distance = code_distance
        self.error_rate = error_rate
        self.num_data_qubits = code_distance ** 2
        self.num_syndrome_bits = 2 * (code_distance ** 2 - 1)
    
    def generate_random_errors(self, batch_size: int) -> np.ndarray:
        """Generate random Pauli X errors"""
        return (np.random.rand(batch_size, self.num_data_qubits) < self.error_rate).astype(int)
    
    def compute_syndromes(self, errors: np.ndarray) -> np.ndarray:
        """
        Compute syndromes from errors.
        
        Simplified model: syndrome bit fires if neighboring qubits have different errors.
        """
        batch_size = errors.shape[0]
        syndromes = np.zeros((batch_size, self.num_syndrome_bits))
        
        # This is a simplified syndrome calculation
        # In practice, use proper stabilizer measurements
        for i in range(batch_size):
            error_pattern = errors[i].reshape(self.code_distance, self.code_distance)
            syndrome_idx = 0
            
            # X-type stabilizers (horizontal edges)
            for row in range(self.code_distance):
                for col in range(self.code_distance - 1):
                    syndromes[i, syndrome_idx] = error_pattern[row, col] ^ error_pattern[row, col + 1]
                    syndrome_idx += 1
            
            # Z-type stabilizers (vertical edges)
            for row in range(self.code_distance - 1):
                for col in range(self.code_distance):
                    syndromes[i, syndrome_idx] = error_pattern[row, col] ^ error_pattern[row + 1, col]
                    syndrome_idx += 1
        
        return syndromes
    
    def generate_dataset(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a dataset of (syndrome, correction) pairs"""
        errors = self.generate_random_errors(num_samples)
        syndromes = self.compute_syndromes(errors)
        
        # For this simplified model, the correction is the error itself
        corrections = errors
        
        return (torch.tensor(syndromes, dtype=torch.float32),
                torch.tensor(corrections, dtype=torch.float32))


# =============================================================================
# Example Usage and Demonstration
# =============================================================================

def main():
    """Demonstrate the BNN QEC decoder"""
    
    print("=" * 70)
    print("Bayesian Neural Network Quantum Error Correction Decoder")
    print("=" * 70)
    
    # Configuration
    code_distance = 3
    data_generator = SurfaceCodeDataGenerator(code_distance=code_distance, error_rate=0.05)
    
    config = DecoderConfig(
        syndrome_size=data_generator.num_syndrome_bits,
        hidden_dims=[128, 64, 32],
        num_samples=50,
        prior_std=1.0,
        posterior_std_init=0.1,
        learning_rate=1e-3,
        kl_weight=1.0 / 1000
    )
    
    print(f"\nCode distance: {code_distance}")
    print(f"Number of data qubits: {data_generator.num_data_qubits}")
    print(f"Number of syndrome bits: {data_generator.num_syndrome_bits}")
    print(f"Network architecture: {config.syndrome_size} -> {' -> '.join(map(str, config.hidden_dims))} -> {config.syndrome_size}")
    
    # Generate training data
    print("\nGenerating training data...")
    train_syndromes, train_corrections = data_generator.generate_dataset(5000)
    train_dataset = torch.utils.data.TensorDataset(train_syndromes, train_corrections)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize decoder
    print("Initializing BNN decoder...")
    decoder = BNNQECDecoder(config)
    
    # Train
    print("\nTraining BNN decoder...")
    decoder.train(train_loader, num_epochs=100, verbose=True)
    
    # Generate test data
    print("\nGenerating test data...")
    test_syndromes, test_corrections = data_generator.generate_dataset(100)
    
    # Evaluate
    print("\nEvaluating decoder...")
    decoder.model.eval()
    
    correct_predictions = 0
    high_uncertainty_cases = 0
    
    for i in range(len(test_syndromes)):
        syndrome = test_syndromes[i].numpy()
        true_correction = test_corrections[i].numpy()
        
        # Decode with uncertainty
        pred_correction, uncertainty = decoder.decode(syndrome, return_uncertainty=True)
        pred_correction = pred_correction.squeeze()
        uncertainty = uncertainty.squeeze()
        
        # Check accuracy
        if np.array_equal(pred_correction, true_correction):
            correct_predictions += 1
        
        # Adaptive decoding
        _, needs_review = decoder.adaptive_decode(syndrome, uncertainty_threshold=0.3)
        if needs_review:
            high_uncertainty_cases += 1
    
    accuracy = correct_predictions / len(test_syndromes)
    print(f"\nTest accuracy: {accuracy:.2%}")
    print(f"High uncertainty cases: {high_uncertainty_cases}/{len(test_syndromes)}")
    
    # Example prediction with uncertainty visualization
    print("\n" + "=" * 70)
    print("Example Prediction with Uncertainty")
    print("=" * 70)
    
    example_syndrome = test_syndromes[0].numpy()
    example_correction, example_uncertainty = decoder.decode(example_syndrome, return_uncertainty=True)
    example_correction = example_correction.squeeze()
    example_uncertainty = example_uncertainty.squeeze()
    
    print(f"\nSyndrome: {example_syndrome}")
    print(f"Predicted correction: {example_correction}")
    print(f"Uncertainty (std dev): {example_uncertainty}")
    print(f"Max uncertainty: {np.max(example_uncertainty):.4f}")
    
    # Plot training history
    print("\nGenerating training plots...")
    fig = decoder.plot_training_history()
    plt.savefig('/mnt/user-data/outputs/bnn_qec_training.png', dpi=150, bbox_inches='tight')
    print("Training plots saved to: bnn_qec_training.png")
    
    # Save model
    torch.save(decoder.model.state_dict(), '/mnt/user-data/outputs/bnn_qec_decoder.pth')
    print("Model saved to: bnn_qec_decoder.pth")
    
    print("\n" + "=" * 70)
    print("BNN QEC Decoder demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
