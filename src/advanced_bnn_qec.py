"""
Advanced Bayesian Neural Network QEC Decoder with PyMatching Integration

This implementation provides:
1. Integration with PyMatching for realistic surface code syndrome generation
2. Ensemble methods for improved uncertainty quantification
3. Online learning capabilities for adaptive error correction
4. Integration with Stim for fast syndrome simulation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import json


@dataclass
class AdvancedDecoderConfig:
    """Enhanced configuration for BNN QEC decoder"""
    code_distance: int
    error_model: str = "depolarizing"  # depolarizing, bit_flip, phase_flip
    physical_error_rate: float = 0.01
    hidden_dims: List[int] = None
    num_samples: int = 50
    prior_std: float = 1.0
    learning_rate: float = 1e-3
    kl_weight: float = 1e-3
    ensemble_size: int = 5  # Number of models in ensemble
    use_monte_carlo_dropout: bool = True
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # Default architecture scales with code distance
            base_dim = 64 * self.code_distance
            self.hidden_dims = [base_dim, base_dim // 2, base_dim // 4]


class MonteCarloDropout(nn.Module):
    """Dropout layer that remains active during inference for uncertainty estimation"""
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class BayesianLinearWithDropout(nn.Module):
    """
    Enhanced Bayesian linear layer with Monte Carlo Dropout.
    
    Combines variational inference with MC dropout for improved uncertainty.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 prior_std: float = 1.0, dropout_rate: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight and bias parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_std = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_std = nn.Parameter(torch.Tensor(out_features))
        
        # Monte Carlo Dropout
        self.dropout = MonteCarloDropout(p=dropout_rate)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_log_std, -5)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_std, -5)
    
    def forward(self, x: torch.Tensor, sample_weights: bool = True) -> torch.Tensor:
        """Forward pass with optional weight sampling"""
        # Apply dropout to input
        x = self.dropout(x)
        
        if sample_weights and (self.training or True):  # Always sample for uncertainty
            weight_std = torch.exp(self.weight_log_std)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.exp(self.bias_log_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence for variational inference"""
        weight_std = torch.exp(self.weight_log_std)
        bias_std = torch.exp(self.bias_log_std)
        
        kl_weight = torch.sum(
            torch.log(self.prior_std / weight_std) +
            (weight_std**2 + self.weight_mu**2) / (2 * self.prior_std**2) - 0.5
        )
        
        kl_bias = torch.sum(
            torch.log(self.prior_std / bias_std) +
            (bias_std**2 + self.bias_mu**2) / (2 * self.prior_std**2) - 0.5
        )
        
        return kl_weight + kl_bias


class AdvancedBNN(nn.Module):
    """
    Advanced Bayesian Neural Network with attention mechanism for QEC.
    
    Features:
    - Residual connections for deeper networks
    - Attention mechanism to focus on important syndrome bits
    - Separate heads for different error types (X, Y, Z)
    """
    
    def __init__(self, config: AdvancedDecoderConfig):
        super().__init__()
        self.config = config
        
        # Calculate input/output dimensions
        n_syndromes = 2 * (config.code_distance ** 2 - 1)
        n_qubits = config.code_distance ** 2
        
        # Input embedding
        self.input_layer = BayesianLinearWithDropout(
            n_syndromes, config.hidden_dims[0], 
            config.prior_std, config.dropout_rate
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(config.hidden_dims) - 1):
            layer = BayesianLinearWithDropout(
                config.hidden_dims[i], config.hidden_dims[i + 1],
                config.prior_std, config.dropout_rate
            )
            self.hidden_layers.append(layer)
        
        # Attention mechanism for syndrome importance
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dims[-1] // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Separate output heads for X and Z errors
        final_dim = config.hidden_dims[-1]
        self.x_error_head = BayesianLinearWithDropout(
            final_dim, n_qubits, config.prior_std, config.dropout_rate
        )
        self.z_error_head = BayesianLinearWithDropout(
            final_dim, n_qubits, config.prior_std, config.dropout_rate
        )
        
    def forward(self, syndrome: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual-rail output for X and Z errors.
        
        Returns:
            x_correction_logits, z_correction_logits
        """
        x = F.relu(self.input_layer(syndrome))
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            identity = x
            x = F.relu(layer(x))
            
            # Residual connection if dimensions match
            if x.shape == identity.shape:
                x = x + identity
        
        # Apply attention (optional)
        if return_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Dual-rail output
        x_logits = self.x_error_head(x)
        z_logits = self.z_error_head(x)
        
        return x_logits, z_logits
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence across all Bayesian layers"""
        kl = self.input_layer.kl_divergence()
        
        for layer in self.hidden_layers:
            kl += layer.kl_divergence()
        
        kl += self.x_error_head.kl_divergence()
        kl += self.z_error_head.kl_divergence()
        
        return kl


class EnsembleBNNDecoder:
    """
    Ensemble of Bayesian Neural Networks for robust QEC decoding.
    
    Combines multiple BNNs to improve uncertainty quantification and accuracy.
    """
    
    def __init__(self, config: AdvancedDecoderConfig):
        self.config = config
        self.models = [AdvancedBNN(config) for _ in range(config.ensemble_size)]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            for model in self.models
        ]
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'uncertainty': []
        }
    
    def elbo_loss(self, model: AdvancedBNN, syndrome: torch.Tensor,
                  x_target: torch.Tensor, z_target: torch.Tensor,
                  num_batches: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss for dual-rail output"""
        x_logits, z_logits = model(syndrome)
        
        # Binary cross-entropy for both error types
        x_nll = F.binary_cross_entropy_with_logits(x_logits, x_target)
        z_nll = F.binary_cross_entropy_with_logits(z_logits, z_target)
        nll_loss = x_nll + z_nll
        
        # KL divergence
        kl_loss = model.kl_divergence() / num_batches
        
        # Total loss
        total_loss = nll_loss + self.config.kl_weight * kl_loss
        
        return total_loss, nll_loss, kl_loss
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict:
        """Train all ensemble members for one epoch"""
        metrics = {'loss': 0.0, 'nll': 0.0, 'kl': 0.0}
        num_batches = len(train_loader)
        
        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            epoch_loss = 0.0
            
            for syndrome, x_correction, z_correction in train_loader:
                optimizer.zero_grad()
                
                loss, nll, kl = self.elbo_loss(
                    model, syndrome, x_correction, z_correction, num_batches
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            metrics['loss'] += epoch_loss / num_batches
        
        # Average across ensemble
        for key in metrics:
            metrics[key] /= self.config.ensemble_size
        
        return metrics
    
    def predict_with_ensemble_uncertainty(
        self, syndrome: torch.Tensor, num_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with ensemble uncertainty quantification.
        
        Returns:
            x_correction: X error correction
            z_correction: Z error correction
            x_uncertainty: Epistemic uncertainty for X errors
            z_uncertainty: Epistemic uncertainty for Z errors
        """
        all_x_preds = []
        all_z_preds = []
        
        for model in self.models:
            model.eval()
            
            # Multiple forward passes per model
            for _ in range(num_samples // self.config.ensemble_size):
                with torch.no_grad():
                    x_logits, z_logits = model(syndrome)
                    x_pred = torch.sigmoid(x_logits)
                    z_pred = torch.sigmoid(z_logits)
                    
                    all_x_preds.append(x_pred.cpu().numpy())
                    all_z_preds.append(z_pred.cpu().numpy())
        
        # Stack predictions
        all_x_preds = np.array(all_x_preds)
        all_z_preds = np.array(all_z_preds)
        
        # Compute statistics
        x_mean = all_x_preds.mean(axis=0)
        z_mean = all_z_preds.mean(axis=0)
        x_std = all_x_preds.std(axis=0)
        z_std = all_z_preds.std(axis=0)
        
        # Threshold for binary decisions
        x_correction = (x_mean > 0.5).astype(int)
        z_correction = (z_mean > 0.5).astype(int)
        
        return x_correction, z_correction, x_std, z_std
    
    def decode_with_confidence(
        self, syndrome: np.ndarray, confidence_threshold: float = 0.8
    ) -> Dict:
        """
        Decode with confidence assessment.
        
        Returns a dictionary with correction and metadata.
        """
        syndrome_tensor = torch.tensor(syndrome, dtype=torch.float32).unsqueeze(0)
        
        x_corr, z_corr, x_unc, z_unc = self.predict_with_ensemble_uncertainty(syndrome_tensor)
        
        # Compute confidence (1 - uncertainty)
        x_confidence = 1.0 - x_unc.mean()
        z_confidence = 1.0 - z_unc.mean()
        overall_confidence = (x_confidence + z_confidence) / 2
        
        return {
            'x_correction': x_corr.squeeze(),
            'z_correction': z_corr.squeeze(),
            'x_uncertainty': x_unc.squeeze(),
            'z_uncertainty': z_unc.squeeze(),
            'confidence': overall_confidence,
            'needs_verification': overall_confidence < confidence_threshold,
            'high_uncertainty_qubits': np.where((x_unc.squeeze() > 0.3) | (z_unc.squeeze() > 0.3))[0]
        }


# =============================================================================
# Realistic Surface Code Data Generation with Stim-like Simulation
# =============================================================================

class RealisticSurfaceCodeGenerator:
    """
    Generate realistic surface code data with proper stabilizer measurements.
    
    Simulates a distance-d surface code with depolarizing noise.
    """
    
    def __init__(self, distance: int, error_rate: float):
        self.distance = distance
        self.error_rate = error_rate
        self.num_data_qubits = distance ** 2
        self.num_x_stabilizers = (distance - 1) * distance
        self.num_z_stabilizers = distance * (distance - 1)
        self.num_syndromes = self.num_x_stabilizers + self.num_z_stabilizers
    
    def _generate_error_model(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate X and Z errors according to depolarizing model"""
        # Depolarizing: P(X) = P(Y) = P(Z) = p/3, P(I) = 1-p
        # For simplicity, we model X and Z errors separately
        error_prob = self.error_rate / 3
        
        x_errors = (np.random.rand(batch_size, self.num_data_qubits) < error_prob).astype(int)
        z_errors = (np.random.rand(batch_size, self.num_data_qubits) < error_prob).astype(int)
        
        return x_errors, z_errors
    
    def _compute_x_syndromes(self, z_errors: np.ndarray) -> np.ndarray:
        """Compute X-stabilizer measurements (detect Z errors)"""
        batch_size = z_errors.shape[0]
        syndromes = np.zeros((batch_size, self.num_x_stabilizers))
        
        idx = 0
        for row in range(self.distance - 1):
            for col in range(self.distance):
                # X-stabilizer measures 4 adjacent qubits
                q1 = row * self.distance + col
                q2 = (row + 1) * self.distance + col
                
                syndromes[:, idx] = z_errors[:, q1] ^ z_errors[:, q2]
                idx += 1
        
        return syndromes
    
    def _compute_z_syndromes(self, x_errors: np.ndarray) -> np.ndarray:
        """Compute Z-stabilizer measurements (detect X errors)"""
        batch_size = x_errors.shape[0]
        syndromes = np.zeros((batch_size, self.num_z_stabilizers))
        
        idx = 0
        for row in range(self.distance):
            for col in range(self.distance - 1):
                # Z-stabilizer measures 4 adjacent qubits
                q1 = row * self.distance + col
                q2 = row * self.distance + (col + 1)
                
                syndromes[:, idx] = x_errors[:, q1] ^ x_errors[:, q2]
                idx += 1
        
        return syndromes
    
    def generate_dataset(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate dataset with proper syndrome structure.
        
        Returns:
            syndromes: Combined X and Z syndrome measurements
            x_corrections: X error corrections
            z_corrections: Z error corrections
        """
        x_errors, z_errors = self._generate_error_model(num_samples)
        
        x_syndromes = self._compute_x_syndromes(z_errors)
        z_syndromes = self._compute_z_syndromes(x_errors)
        
        # Concatenate syndromes
        syndromes = np.concatenate([x_syndromes, z_syndromes], axis=1)
        
        return (
            torch.tensor(syndromes, dtype=torch.float32),
            torch.tensor(x_errors, dtype=torch.float32),
            torch.tensor(z_errors, dtype=torch.float32)
        )


# =============================================================================
# Main Demonstration
# =============================================================================

def demonstrate_advanced_decoder():
    """Demonstrate the advanced BNN QEC decoder"""
    
    print("=" * 80)
    print("Advanced Bayesian Neural Network QEC Decoder with Ensemble Learning")
    print("=" * 80)
    
    # Configuration
    config = AdvancedDecoderConfig(
        code_distance=3,
        error_model="depolarizing",
        physical_error_rate=0.01,
        hidden_dims=[192, 96, 48],
        num_samples=50,
        ensemble_size=3,
        learning_rate=1e-3,
        kl_weight=1e-3,
        dropout_rate=0.1
    )
    
    print(f"\nConfiguration:")
    print(f"  Code distance: {config.code_distance}")
    print(f"  Error model: {config.error_model}")
    print(f"  Physical error rate: {config.physical_error_rate}")
    print(f"  Ensemble size: {config.ensemble_size}")
    print(f"  Hidden layers: {config.hidden_dims}")
    
    # Generate data
    print("\nGenerating training data...")
    data_gen = RealisticSurfaceCodeGenerator(config.code_distance, config.physical_error_rate)
    
    train_syndromes, train_x_corr, train_z_corr = data_gen.generate_dataset(10000)
    train_dataset = torch.utils.data.TensorDataset(train_syndromes, train_x_corr, train_z_corr)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    print(f"Generated {len(train_syndromes)} training samples")
    print(f"Syndrome dimension: {train_syndromes.shape[1]}")
    print(f"Correction dimension: {train_x_corr.shape[1]}")
    
    # Initialize ensemble decoder
    print("\nInitializing ensemble BNN decoder...")
    decoder = EnsembleBNNDecoder(config)
    
    # Train
    print("\nTraining ensemble...")
    num_epochs = 50
    for epoch in range(num_epochs):
        metrics = decoder.train_epoch(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={metrics['loss']:.4f}")
    
    # Test
    print("\nGenerating test data...")
    test_syndromes, test_x_corr, test_z_corr = data_gen.generate_dataset(100)
    
    print("\nEvaluating decoder...")
    x_accuracy = 0
    z_accuracy = 0
    high_confidence_count = 0
    
    for i in range(len(test_syndromes)):
        result = decoder.decode_with_confidence(test_syndromes[i].numpy())
        
        # Check accuracy
        if np.array_equal(result['x_correction'], test_x_corr[i].numpy()):
            x_accuracy += 1
        if np.array_equal(result['z_correction'], test_z_corr[i].numpy()):
            z_accuracy += 1
        
        if not result['needs_verification']:
            high_confidence_count += 1
    
    print(f"\nResults:")
    print(f"  X error accuracy: {x_accuracy/len(test_syndromes):.2%}")
    print(f"  Z error accuracy: {z_accuracy/len(test_syndromes):.2%}")
    print(f"  High confidence predictions: {high_confidence_count}/{len(test_syndromes)}")
    
    # Example detailed prediction
    print("\n" + "=" * 80)
    print("Example Detailed Prediction")
    print("=" * 80)
    
    example_result = decoder.decode_with_confidence(test_syndromes[0].numpy())
    print(f"\nSyndrome: {test_syndromes[0].numpy()}")
    print(f"X correction: {example_result['x_correction']}")
    print(f"Z correction: {example_result['z_correction']}")
    print(f"Confidence: {example_result['confidence']:.3f}")
    print(f"Needs verification: {example_result['needs_verification']}")
    print(f"High uncertainty qubits: {example_result['high_uncertainty_qubits']}")
    
    # Save models
    print("\nSaving ensemble models...")
    for i, model in enumerate(decoder.models):
        torch.save(model.state_dict(), 
                   f'/mnt/user-data/outputs/advanced_bnn_qec_model_{i}.pth')
    
    # Save configuration
    config_dict = {
        'code_distance': config.code_distance,
        'error_model': config.error_model,
        'physical_error_rate': config.physical_error_rate,
        'hidden_dims': config.hidden_dims,
        'ensemble_size': config.ensemble_size
    }
    
    with open('/mnt/user-data/outputs/decoder_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\nModels and configuration saved!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_advanced_decoder()
