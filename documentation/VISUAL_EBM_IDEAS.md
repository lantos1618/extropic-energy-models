# Visually Striking Energy-Based Models for THRML

## What Makes a Good Visual EBM?

**Requirements:**
1. Native energy formulation (not forced)
2. Spatial structure (2D grid or network)
3. Emergent patterns (domains, clusters, waves)
4. Phase transitions or critical phenomena
5. Dynamic evolution you can watch

---

## 🎨 Category 1: Spin Systems (Like We Did)

### ✅ Already Implemented:
- **Ising Model** (2-state) - Ferromagnetic domains
- **Potts Model** (q-state) - Multi-color domains

### 🔥 Could Add:

**1. XY Model (Continuous Spins)**
```
Energy: E = -J Σ cos(θ_i - θ_j)
Spins: θ ∈ [0, 2π] (directions, like compass needles)
Visual: Arrow field showing spin directions
Phase: Kosterlitz-Thouless transition (vortices!)
```
**Why cool:** Vortex-antivortex pairs form, swirl patterns

**2. Heisenberg Model (3D Spins)**
```
Energy: E = -J Σ S_i · S_j
Spins: S = (sx, sy, sz) on unit sphere
Visual: 3D vector field, color by direction
Phase: True long-range order in 3D
```
**Why cool:** Complex magnetic textures, skyrmions

**3. Clock Model (q discrete angles)**
```
Energy: E = -J Σ cos(θ_i - θ_j)
Spins: θ ∈ {0, 2π/q, 4π/q, ...}
Visual: Like Potts but with angle alignment
Phase: Multiple transitions for q≥5
```
**Why cool:** Intermediate between Ising and XY

---

## 🧩 Category 2: Optimization Problems (Graph-Based)

### 4. Graph Coloring
```
Energy: E = Σ penalty if adjacent nodes same color
States: Each node has color c ∈ {1,...,k}
Visual: Network with colored nodes
Goal: Find valid coloring minimizing colors
```
**Why cool:** Watch colors spread and compete for territory
**Example:** Map coloring, scheduling conflicts

### 5. Max-Cut Problem
```
Energy: E = -Σ (s_i ≠ s_j) over edges
States: Binary partition {-1, +1}
Visual: Network with two colors trying to maximize edge cuts
Goal: Partition graph to maximize cuts
```
**Why cool:** Clusters form trying to maximize boundaries
**Applications:** Circuit design, community detection

### 6. Traveling Salesman (TSP)
```
Energy: E = total path length
States: Permutations of cities
Visual: Path through cities evolving
Goal: Find shortest tour
```
**Why cool:** Watch path untangle itself
**Challenge:** Large state space, need clever encoding

### 7. K-Means Clustering (Continuous)
```
Energy: E = Σ ||x_i - μ_cluster(i)||²
States: Cluster assignments
Visual: Points colored by cluster, centers moving
Goal: Minimize within-cluster variance
```
**Why cool:** Voronoi regions form and shift

---

## 🌊 Category 3: Reaction-Diffusion & Pattern Formation

### 8. Cahn-Hilliard Equation (Phase Separation)
```
Energy: E = ∫ [ε²|∇φ|² + ψ(φ)] dx
States: Continuous field φ(x) ∈ [-1, 1]
Visual: Domains coarsening, spinodal decomposition
Phase: Separates into +1 and -1 regions
```
**Why cool:** Beautiful domain coarsening patterns
**Applications:** Alloy formation, biological membranes

### 9. Voter Model (Opinion Dynamics)
```
Energy: E = -Σ (s_i = s_j) over edges
States: Binary opinions {0, 1}
Visual: Spatial patterns of agreement/disagreement
Dynamics: Nodes copy neighbors' opinions
```
**Why cool:** Consensus formation, domain walls
**Applications:** Social networks, epidemics

### 10. Cellular Automata as EBM
```
Energy: E = -Σ f(local_neighborhood)
States: Grid of cells with rules
Visual: Conway's Game of Life, but energy-based
Example: Majority rule, totalistic rules
```
**Why cool:** Complex from simple rules

---

## 🎲 Category 4: Probabilistic & Generative

### 11. Restricted Boltzmann Machine (RBM)
```
Energy: E = -Σ w_ij v_i h_j - Σ b_i v_i - Σ c_j h_j
States: Visible v and hidden h layers
Visual: Learn patterns from data (MNIST digits)
Goal: Generative model
```
**Why cool:** Watch it learn to generate digits
**Applications:** Unsupervised learning, deep belief nets

### 12. Hopfield Network (Associative Memory)
```
Energy: E = -½ Σ w_ij s_i s_j
States: Binary neurons
Visual: Watch network recall patterns from noise
Goal: Store and retrieve memories
```
**Why cool:** Pattern completion, attractor dynamics
**Demo:** Store images, corrupt them, watch recovery

### 13. Energy-Based GAN Discriminator
```
Energy: E(x) = discriminator score
States: Images (continuous)
Visual: Generate images that fool discriminator
Goal: Match data distribution
```
**Why cool:** Generate realistic images
**Challenge:** High-dimensional, needs GPU

---

## 🏔️ Category 5: Physical Simulations

### 14. Lattice Gas (Fluid Flow)
```
Energy: E = kinetic + interaction
States: Particles on lattice sites
Visual: Fluid flowing around obstacles
Phase: Gas/liquid transition
```
**Why cool:** Emergent fluid behavior from simple rules

### 15. Dimer Model (Tiling)
```
Energy: E = -# of valid tilings
States: Domino tilings of grid
Visual: Perfect matchings, Arctic circle
Phase: Ordered/disordered boundary
```
**Why cool:** Beautiful arctic circle phenomenon

### 16. Vertex Model (Biological Tissues)
```
Energy: E = Σ (A - A₀)² + Σ P_perimeter
States: Cell shapes in tissue
Visual: Cells rearranging to minimize energy
Phase: Fluid-to-solid transitions
```
**Why cool:** Models epithelial tissues, cell sorting

---

## 🌟 Category 6: Exotic & Fun

### 17. Sandpile Model (Self-Organized Criticality)
```
Energy: E = Σ h_i (height at site i)
States: Integer heights
Visual: Avalanches cascading
Phase: Critical state (no tuning needed!)
```
**Why cool:** Power-law avalanches, no parameters
**Famous:** Bak-Tang-Wiesenfeld model

### 18. Loop Erased Random Walk (LERW)
```
Energy: E = -# of paths
States: Self-avoiding paths
Visual: Random walks that erase loops
Phase: Fractal dimension
```
**Why cool:** Beautiful fractal patterns

### 19. Spin Glass (Frustration)
```
Energy: E = -Σ J_ij s_i s_j (J random ±1)
States: Spins on lattice
Visual: Frustrated interactions, no global order
Phase: Rugged energy landscape
```
**Why cool:** Glassy behavior, metastable states
**Applications:** Optimization, neural networks

### 20. Forest Fire Model
```
Energy: E = connectivity measure
States: Empty/tree/burning
Visual: Trees growing, fires spreading
Phase: Percolation transition
```
**Why cool:** Spatiotemporal patterns, clustering

---

## 🎯 My Top 5 Recommendations

### Most Visually Striking + THRML-Compatible:

**1. XY Model** ⭐⭐⭐⭐⭐
- **Why:** Vortices are gorgeous, continuous spins
- **Challenge:** Need continuous sampling (but THRML might support)
- **Visual:** Arrow field with swirling vortices

**2. Graph Coloring** ⭐⭐⭐⭐⭐
- **Why:** Great for networks, clear optimization goal
- **Challenge:** Easy to implement
- **Visual:** Colors spreading and competing

**3. RBM on MNIST** ⭐⭐⭐⭐
- **Why:** Learn to generate handwritten digits
- **Challenge:** Need training data
- **Visual:** Watch digits emerge from noise

**4. Voter Model** ⭐⭐⭐⭐
- **Why:** Simple, beautiful spatial patterns
- **Challenge:** Very easy
- **Visual:** Opinion domains forming

**5. Sandpile** ⭐⭐⭐⭐⭐
- **Why:** Self-organized criticality is mind-blowing
- **Challenge:** Might need custom dynamics
- **Visual:** Cascading avalanches

---

## 🚀 Implementation Difficulty

### Easy (Can do now):
- ✅ Voter Model
- ✅ Graph Coloring
- ✅ Max-Cut
- ✅ Clock Model

### Medium (Need some work):
- ⚠️ XY Model (continuous sampling)
- ⚠️ RBM (need data + training)
- ⚠️ Hopfield Network (pattern storage)
- ⚠️ Sandpile (custom rules)

### Hard (Research project):
- ❌ Energy-based GAN (deep learning)
- ❌ Vertex Model (complex geometry)
- ❌ Lattice Gas (fluid dynamics)

---

## 🎨 Visualization Appeal

### Most Photogenic:

1. **XY Model vortices** - Swirling patterns, topological
2. **Sandpile avalanches** - Cascading colors, fractal
3. **RBM digits** - Generated images
4. **Graph coloring** - Network art
5. **Cahn-Hilliard** - Domain coarsening

### Best for Animation:

1. **Voter Model** - Domains growing/shrinking
2. **Forest Fire** - Spreading and regrowth
3. **Ising/Potts** - Phase transitions
4. **Sandpile** - Avalanche cascades
5. **Hopfield** - Memory recall

---

## 💡 What Should We Build Next?

### Option A: XY Model (Vortices!)
```python
# Continuous spins on circle
θ_i ∈ [0, 2π]
E = -J Σ cos(θ_i - θ_j)
Visual: Arrow field, color by angle
See: Vortex-antivortex pairs
```

### Option B: Graph Coloring
```python
# Network with k colors
c_i ∈ {1, 2, ..., k}
E = Σ penalty(c_i = c_j)
Visual: Network with colored nodes
See: Color domains spreading
```

### Option C: Voter Model
```python
# Opinion dynamics
s_i ∈ {0, 1}
E = -Σ (s_i = s_j)
Visual: Spatial agreement patterns
See: Consensus formation
```

### Option D: Sandpile
```python
# Self-organized criticality
h_i ∈ {0, 1, 2, 3}
Rule: If h_i ≥ 4, topple
Visual: Avalanche cascades
See: Power-law behavior
```

---

## 🎓 Educational Value

**Best for teaching energy-based computing:**
1. Ising/Potts (already have) ✅
2. Graph Coloring (combinatorial optimization)
3. RBM (machine learning connection)
4. Voter Model (opinion dynamics)
5. Max-Cut (graph optimization)

**Best for "wow factor":**
1. XY Model vortices
2. RBM generating digits
3. Sandpile avalanches
4. Hopfield memory recall
5. Cahn-Hilliard phase separation

---

## 🤔 What Do You Want?

**Pick a direction:**
- 🎨 **Visual beauty?** → XY Model or Sandpile
- 🧮 **Practical optimization?** → Graph Coloring or Max-Cut
- 🧠 **Machine learning?** → RBM or Hopfield
- 📊 **Simple but effective?** → Voter Model
- 🌀 **Physics-y?** → XY, Heisenberg, or Cahn-Hilliard

**All of these are:**
- ✅ Native energy-based problems
- ✅ Not circular/performative
- ✅ Visually interesting
- ✅ Can use THRML (or similar tools)
- ✅ Have real applications

Let me know which sounds coolest and I'll implement it!
