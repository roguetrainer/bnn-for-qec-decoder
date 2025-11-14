
# Primary Sources for This Implementation

The code we created is **not directly based on any single GitHub repository**. Instead, it's a **synthesis of concepts and techniques** from multiple academic papers and standard Bayesian deep learning practices.

### Key Influences (in order of importance):

#### 1. **Standard Bayesian Neural Network Literature**
The core BNN implementation (variational inference, reparameterization trick, ELBO loss) comes from foundational papers:
- **Blundell et al. (2015)**: "Weight Uncertainty in Neural Networks" - The classic Bayes by Backprop paper
- **Gal & Ghahramani (2016)**: "Dropout as a Bayesian Approximation" - MC Dropout
- Standard PyTorch/TensorFlow BNN tutorials and examples

This is the **main source** - the BNN implementation follows standard practices from the Bayesian deep learning community, not any specific QEC repository.

#### 2. **QEC Decoder Papers (Conceptual Framework)**
For understanding how to apply BNNs to QEC:
- **QuBA/SAGU (Mi & Mueller, 2025)** - arXiv:2510.06257: Conceptual framework for Bayesian GNN decoders
- **Torlai & Melko (2017)** - Scientific Reports: Neural probabilistic decoders for stabilizer codes
- **Varsamopoulos et al. (2018)**: Feedforward neural networks for surface codes

These papers provided the **problem formulation** (syndrome → correction mapping) but not the actual code.

#### 3. **Surface Code Basics**
The simplified surface code syndrome generation is based on standard QEC textbooks:
- **Nielsen & Chuang**: Quantum Computation and Quantum Information
- Standard surface code descriptions in QEC literature

### What About GraphQEC?

The repository (https://github.com/adelshb/graphqec) is related to recent work:

**GraphQEC (2025)** - arXiv:2502.19971: "Efficient and Universal Neural-Network Decoder for Stabilizer-Based Quantum Error Correction"

We cited this paper in our overview document as one of the recent advances, but we did not use their code as a basis because:

1. **Different architecture focus**: GraphQEC uses temporal GNNs with graph structures, while our implementation demonstrates core BNN concepts with simpler feedforward architectures
2. **Educational purpose**: our code is designed to be pedagogical - showing BNN fundamentals from scratch
3. **Self-contained**: we wanted to create a standalone implementation that doesn't require external QEC frameworks

### What Our Code Actually Implements

```python
# Our implementation stack:
BayesianLinear layer
├─ Based on: Blundell et al. (2015) Bayes by Backprop
├─ Standard variational inference
└─ Reparameterization trick

SimpleBNN / AdvancedBNN
├─ Based on: Standard feedforward/ensemble BNN architectures
├─ Not from any specific QEC repo
└─ Demonstrates core concepts clearly

Surface Code Data Generation
├─ Based on: Textbook surface code definitions
├─ Simplified for demonstration
└─ NOT production-quality syndrome generation

Training & Inference
├─ Standard PyTorch training loops
├─ ELBO loss (standard in BNN literature)
└─ Uncertainty quantification via sampling
```

### Comparison with GraphQEC

| Aspect | Our Implementation | GraphQEC Repository |
|--------|------------------|---------------------|
| **Architecture** | Feedforward BNN | Temporal GNN |
| **Focus** | Bayesian inference fundamentals | Graph-based message passing |
| **Purpose** | Educational + demonstration | Research/production |
| **Dependencies** | Pure PyTorch | Likely requires QEC frameworks |
| **Code structure** | From scratch, pedagogical | Optimized, complex |
| **Source** | Standard BNN literature | Novel research architecture |

### Other Notable QEC Decoder Repositories

While we didn't base our code on them, these are important in the ecosystem:

1. **PyMatching** - Classical MWPM decoder (not neural)
2. **Stim** - Fast quantum error correction simulator
3. **Qiskit QEC modules** - Various decoder implementations
4. **DeepNeuralDecoder** (Chamberland & Ronagh) - Early neural decoder work

None of these were direct code sources for our implementation.

### The "Original Contribution"

What we created is:
- **Educational reimplementation** of BNN concepts applied to QEC
- **Synthesis** of ideas from multiple papers into working code
- **Not claiming novelty** in algorithms, but in presentation and accessibility
- **Pedagogical focus**: Clear, documented, runnable examples

### If You Want Production-Quality Code

For research or production use, you should look at:

1. **GraphQEC** (if that repo is public) - Latest temporal GNN methods
2. **AlphaQubit** - If Google releases code (Nature 2024 paper)
3. **PyMatching** + your own neural decoder - Hybrid approach
4. **Stim + custom decoder** - For fast syndrome generation

### The Honest Answer

Our code is **"inspired by the literature but implemented from scratch"** - it's a teaching implementation that demonstrates concepts from papers like QuBA, but uses standard BNN building blocks from the general Bayesian ML community.


The key distinction: our code teaches BNN concepts for QEC. Research repos like GraphQEC push state-of-the-art performance. Both are valuable, but serve different purposes! 🎯