# Knowledge Notes: Quantum Reservoir Computing with PCA-Encoded Inputs

## Corpus Overview

- Total sources: 37
- Primary papers/reports: 30
- Sources from 2023+: 27

## Seed-Paper Technical Highlights

### arxiv:2509.06873 | Entanglement and Classical Simulability in Quantum Extreme Learning Machines

- Why it matters: Quantum Machine Learning (QML) has emerged as a promising framework to exploit quantum mechanics for computational advantage. Here we investigate Quantum Extreme Learning Machines (QELMs), a quantum analogue of classical Extreme Learning Machines in which training is restricted to the output layer. Our architecture combines dimensionality reduction (via PCA or Autoencoders), quantum state encoding, evolution under an XX Hamiltonian, and projective measurement to produce features for a classical single-layer classifier. By analyzing the classification accuracy as a function of evolution time, we identify a sharp transition between low- and high-accuracy regimes, followed by saturation. Remarkably, the saturation value coincides with that obtained using random unitaries that generate maximally complex dynamics, even though the XX model is integrable and local. We show that this performance enhancement correlates with the onset of entanglement, which improves the embedding of classical data in Hilbert space and leads to more separable clusters in measurement probability space. Thus, entanglement contributes positively to the structure of the data embedding, improving learnability without necessarily implying computational advantage. For the image classification tasks studied in this work (namely MNIST, Fashion-MNIST, and CIFAR-10) the required evolution time corresponds to information exchange among nearest neighbors and is independent of the system size. This implies that QELMs rely on limited entanglement and remain classically simulable for a broad class of learning problems. Our results clarify how moderate quantum correlations bridge the gap between quantum dynamics and classical feature learning.

- Key equation/definition: `|ψ⟩ = cos( ) |0⟩ + eiϕ sin( ) |1⟩`

- Central claim: Riera2  Institut de Fı́sica d’Altes Energies (IFAE) - The Barcelona Institute of Science and Technology (BIST), Campus UAB, 08193 Bellaterra (Barcelona), Spain 2 Qilimanjaro Quantum Tech S.L., Carrer de Veneçuela, 74, 08019 Barcelona, Spain 3 Departament de Fı́sica, Universitat Autònoma de Barcelona 4 Dipartimento di Fisica, Università della Calabria, 87036 Arcavacata di Rende (CS), Italy 5 INFN, gruppo collegato di Cosenza, 87036 Arcavacata di Rende (CS), Italy * E-mail: adelorenzis@ifae.es  Abstract Quantum Machine Learning (QML) has emerged as a promising framework to exploit quantum mechanics for computational advantage

- Limitation: Feature reduction: One of the main challenges in applying quantum machine learning lies in the limited number of qubits currently available in quantum hardware

### arxiv:2409.00998 | Harnessing Quantum Extreme Learning Machines for image classification

- Why it matters: Interest in quantum machine learning is increasingly growing due to its potential to offer more efficient solutions for problems that are difficult to tackle with classical methods. In this context, the research work presented here focuses on the use of quantum machine learning techniques for image classification tasks. We exploit a quantum extreme learning machine by taking advantage of its rich feature map provided by the quantum reservoir substrate. We systematically analyse different phases of the quantum extreme learning machine process, from the dataset preparation to the image final classification. In particular, we have tested different encodings, together with Principal Component Analysis, the use of Auto-Encoders, as well as the dynamics of the model through the use of different Hamiltonians for the quantum reservoir. Our results show that the introduction of a quantum reservoir systematically improves the accuracy of the classifier. Additionally, while different encodings can lead to significantly different performances, Hamiltonians with varying degrees of connectivity exhibit the same discrimination rate, provided they are interacting.

- Key equation/definition: `|ψ⟩ = cos( ) |0⟩ + eiϕ sin( ) |1⟩`

- Central claim: We exploit a quantum extreme learning machine by taking advantage of its rich feature map provided by the quantum reservoir substrate

- Limitation: Only the weights of this layer are optimized, which significantly reduces training complexity and computational cost

### arxiv:2602.14677 | Kernel-based optimization of measurement operators for quantum reservoir computers

- Why it matters: Finding optimal measurement operators is crucial for the performance of quantum reservoir computers (QRCs), since they employ a fixed quantum feature map. We formulate the training of both stateless (quantum extreme learning machines, QELMs) and stateful (memory dependent) QRCs in the framework of kernel ridge regression. We thus extend the kernel viewpoint of supervised quantum models to recurrent QRCs by deriving an exact Hilbert--Schmidt kernel representation of the optimal readout observable on history space. This approach renders an optimal measurement operator that minimizes prediction error for a given reservoir and training dataset. For large qubit numbers, this method is more efficient than the conventional training of QRCs. We discuss efficiency and practical implementation strategies, including Pauli basis decomposition and operator diagonalization, to adapt the optimal observable to hardware constraints. To demonstrate the effectiveness of this approach, we present numerical experiments on image classification and time series prediction tasks, including chaotic and strongly non-Markovian systems. The developed method can also be applied to other quantum machine learning models.

- Key equation/definition: `Hilbert space dimension (D = 2N )`

- Central claim: Due to the training of the internal layers, this usually does not pose a severe problem for the accuracy of the model

- Limitation: Finding optimal measurement operators is thus a key challenge in QRC design [15–19]

### arxiv:2008.08605 | The effect of data encoding on the expressive power of variational quantum machine learning models

- Why it matters: Quantum computers can be used for supervised learning by treating parametrised quantum circuits as models that map data inputs to predictions. While a lot of work has been done to investigate practical implications of this approach, many important theoretical properties of these models remain unknown. Here we investigate how the strategy with which data is encoded into the model influences the expressive power of parametrised quantum circuits as function approximators. We show that one can naturally write a quantum model as a partial Fourier series in the data, where the accessible frequencies are determined by the nature of the data encoding gates in the circuit. By repeating simple data encoding gates multiple times, quantum models can access increasingly rich frequency spectra. We show that there exist quantum models which can realise all possible sets of Fourier coefficients, and therefore, if the accessible frequency spectrum is asymptotically rich enough, such models are universal function approximators.

- Key equation/definition: `encode data inputs x = (x1 , . . . , xN ) as well as trainable weights θ = (θ1 , . . . , θM ). The circuit is measured`

- Central claim: We show that one can naturally write a quantum model as a partial Fourier series in the data, where the accessible frequencies are determined by the nature of the data encoding gates in the circuit

- Limitation: Repeated Pauli encodings linearly extend the frequency spectrum  Given the severe limitations exposed in the previous section, a natural question is how we can extend the accessible frequency spectrum of a quantum model

### arxiv:2101.11020 | Supervised quantum machine learning models are kernel methods

- Why it matters: With near-term quantum devices available and the race for fault-tolerant quantum computers in full swing, researchers became interested in the question of what happens if we replace a supervised machine learning model with a quantum circuit. While such "quantum models" are sometimes called "quantum neural networks", it has been repeatedly noted that their mathematical structure is actually much more closely related to kernel methods: they analyse data in high-dimensional Hilbert spaces to which we only have access through inner products revealed by measurements. This technical manuscript summarises and extends the idea of systematically rephrasing supervised quantum models as a kernel method. With this, a lot of near-term and fault-tolerant quantum models can be replaced by a general support vector machine whose kernel computes distances between data-encoding quantum states. Kernel-based training is then guaranteed to find better or equally good quantum models than variational circuit training. Overall, the kernel perspective of quantum machine learning tells us that the way that data is encoded into quantum states is the main ingredient that can potentially set quantum models apart from classical machine learning models.

- Key equation/definition: `ρ(x) = ∣φ(x)⟩⟨φ(x)∣ as the feature “vectors”2 instead of the Dirac vectors ∣φ(x)⟩ (see Section V A). This was first`

- Central claim: The advantages of kernel-based training are therefore that we are guaranteed to find the globally optimal measurement over all possible quantum models

- Limitation: Training a quantum model is the problem of finding the measurement that minimises a data-dependent cost function

### arxiv:2407.02553 | Large-scale quantum reservoir learning with an analog quantum computer

- Why it matters: Quantum machine learning has gained considerable attention as quantum technology advances, presenting a promising approach for efficiently learning complex data patterns. Despite this promise, most contemporary quantum methods require significant resources for variational parameter optimization and face issues with vanishing gradients, leading to experiments that are either limited in scale or lack potential for quantum advantage. To address this, we develop a general-purpose, gradient-free, and scalable quantum reservoir learning algorithm that harnesses the quantum dynamics of neutral-atom analog quantum computers to process data. We experimentally implement the algorithm, achieving competitive performance across various categories of machine learning tasks, including binary and multi-class classification, as well as timeseries prediction. Effective and improving learning is observed with increasing system sizes of up to 108 qubits, demonstrating the largest quantum machine learning experiment to date. We further observe comparative quantum kernel advantage in learning tasks by constructing synthetic datasets based on the geometric differences between generated quantum and classical data kernels. Our findings demonstrate the potential of utilizing classically intractable quantum correlations for effective machine learning. We expect these results to stimulate further extensions to different quantum hardware and machine learning paradigms, including early fault-tolerant hardware and generative machine learning tasks.

- Key equation/definition: `Rydberg state of an atom (|rj ⟩), nj = |rj ⟩ ⟨rj |, while`

- Central claim: Despite this promise, most contemporary quantum methods require significant resources for variational parameter optimization and face issues with vanishing gradients, leading to experiments that are either limited in scale or lack potential for quantum advantage

- Limitation: To address this, we develop a general-purpose, gradient-free, and scalable quantum reservoir learning algorithm that harnesses the quantum dynamics of neutral-atom analog quantum computers to process data

### arxiv:2412.06758 | Robust Quantum Reservoir Computing for Molecular Property Prediction

- Why it matters: Machine learning has been increasingly utilized in the field of biomedical research to accelerate the drug discovery process. In recent years, the emergence of quantum computing has been followed by extensive exploration of quantum machine learning algorithms. Quantum variational machine learning algorithms are currently the most prevalent but face issues with trainability due to vanishing gradients. An emerging alternative is the quantum reservoir computing (QRC) approach, in which the quantum algorithm does not require gradient evaluation on quantum hardware. Motivated by the potential advantages of the QRC method, we apply it to predict the biological activity of potential drug molecules based on molecular descriptors. We observe more robust QRC performance as the size of the dataset decreases, compared to standard classical models, a quality of potential interest for pharmaceutical datasets of limited size. In addition, we leverage the uniform manifold approximation and projection technique to analyze structural changes as classical features are transformed through quantum dynamics and find that quantum reservoir embeddings appear to be more interpretable in lower dimensions.

- Key equation/definition: `between a ground (|gj ⟩, j indexes atoms) and an excited state of an atom (|rj ⟩), nj = |rj ⟩ ⟨rj |, while`

- Central claim: Motivated by the potential advantages of the QRC method, we apply it to predict the biological activity of potential drug molecules based on molecular descriptors

- Limitation: However, an immediate challenge is how to extract the relevant molecular features to improve the overall performance of the model, since the structure-property relationship for molecules is complex [1, 3, 10]

## Cross-Paper Similarities

- Encoding-first view: multiple QELM/QRC papers and data-encoding analyses indicate feature quality is dominated by encoding choices before readout optimization.

- Readout bottleneck: both classical RC and QRC families rely on linear/ridge readouts, making fair comparison highly sensitive to regularization and state dimensionality.

- Entanglement nuance: recent QELM papers link gains to regimes where induced correlations are not trivially classically simulable.

## Cross-Paper Differences

- Reservoir substrates differ (spin chains, photonics, neutral atoms, superconducting circuits), which changes achievable observables and effective kernels.

- Benchmarks vary from easy image datasets to chaotic time-series; reported gains are often domain-dependent.

- Measurement design differs (fixed Pauli subsets vs optimized operators), materially affecting downstream separability.

## Dataset Implications for This Project

- MNIST alone is likely insufficient for robust quantum-advantage conclusions after PCA.

- Prefer at least one harder benchmark (CIFAR-10/Fashion-MNIST/EMNIST/Kuzushiji-MNIST) under matched PCA dimensionality and class balance.

## Reproducibility Checklist Inputs

- Fix random seeds for PCA, reservoir initialization, and readout training.

- Match preprocessing and readout solver across quantum and classical reservoirs.

- Report compute budget compliance under CPU-only constraints.

## Retry Addendum (2026-03-23)

- Added 7 newly discovered sources via arXiv API and direct-fetch URL validation.
- Additional coverage extends QRC literature to photonic hybrid forecasting, autoencoding, feedback-based optical reservoirs, Wiener-structured QRC, and image denoising.
- Added DOI-backed general QML framing sources for stronger baseline/claim hygiene (Nature 2019, Nature 2017).

### Newly Added Source IDs
- https://arxiv.org/abs/2603.10707: Hybrid Photonic Quantum Reservoir Computing for High-Dimensional Financial Surface Prediction (2026)
- https://arxiv.org/abs/2602.19700: Quantum Reservoir Autoencoder: Conditions, Protocol, and Noise Resilience (2026)
- https://arxiv.org/abs/2602.17440: A Programmable Linear Optical Quantum Reservoir with Measurement Feedback for Time Series Analysis (2026)
- https://arxiv.org/abs/2601.04812: Quantum Wiener architecture for quantum reservoir computing (2026)
- https://arxiv.org/abs/2512.18612: Image Denoising via Quantum Reservoir Computing (2025)
- https://www.nature.com/articles/s41586-019-0980-2: Supervised learning with quantum-enhanced feature spaces (2019)
- https://www.nature.com/articles/nature23474: Quantum machine learning (2017)

### Retry-Specific Implications
- New time-series and denoising sources reinforce that evaluation should include non-MNIST tasks with matched preprocessing and regularization.
- Measurement design and feedback loops remain central confounders when attributing gains to quantum dynamics alone.

## Retry Validation Addendum (2026-03-23T02:56:40Z)

- Recovery retry now uses the 44-source corpus artifact with depth coverage populated across the primary paper set in the phase payload.
- Semantic expectations are explicitly set to `ka_source_coverage` and `ka_traceability`, with matching `execution_report.expectation_results` evidence links to `payload.sources`, `knowledge/refs.jsonl`, `knowledge/notes.md`, and `phase_outputs/research_trace.json`.
- No additional source append was required in this retry because the corpus already exceeded recovery thresholds (`total=44`, `primary=37`, `recent_2023_plus=32`) from the preceding discovery pass.
