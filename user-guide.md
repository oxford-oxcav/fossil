# Fossil User Guide

The `CegisConfig` class is a central configuration component. In this guide, each parameter within the `CegisConfig` class is detailed in its own numbered section to provide a comprehensive understanding.

---

## 1. SYSTEM

**Description:** Represents the system being used.

**Default Value:** `Must be specified`

---

## 2. CERTIFICATE

**Description:** Specifies the type of certificate to be used.

**Options:** LYAPUNOV, BARRIER, BARRIERALT, ROA, RWA, RSWA, RWS, RSWS, STABLESAFE, RAR

**Default Value:**  `Must be specified`

---

## 3. DOMAINS

**Description:** Dictionary that defines different domains for the chosen certificate. See Set-guide.md for more information.

**Default Value:** `Must be specified`

---

## 4. N_DATA

**Description:** Number of data points to be sampled for each domain. Can be specified as a dictionary (specified by domain) or a single integer (for all domains). See Set-guide.md for more information.

**Default Value:** `500`

---

## 5. SYMMETRIC_BELT

**Description:** A boolean parameter for barrier synthesis. If true, $\dot{B} <= 0 $ are considered when training the Lie condition. If false, $- \epsilon < \dot{B} < \epsilon $ are considered where $\epsilon$ is a small positive number.

**Default Value:** `False`

---

## 6. CEGIS_MAX_ITERS

**Description:** Specifies the maximum iterations for CEGIS.

**Default Value:** `10`

---

## 7. CEGIS_MAX_TIME_S

**Description:** Specifies the maximum time in seconds for CEGIS.

**Default Value:** `math.inf`

---

## 8. TIME_DOMAIN

**Description:** Specifies the time domain of the specified dynamical model. Discrete time models support only Lyapunov and BarrierAlt certificates. Continuous time models support all certificates.

**Options:** CONTINUOUS, DISCRETE

**Default Value:** `CONTINUOUS`

---

## 10. VERIFIER

**Description:** Specifies the type of verifier to be used.

**Options:** Z3, DREAL

**Default Value:** `Z3`

---

## 14. LEARNING_RATE

**Description:** Learning rate used for training the neural network in PyTorch.

**Default Value:** `0.1`

---

## 15. FACTORS

**Description:** Specifies the factors considered during learning.

**Options:** QUADRATIC, NONE

**Default Value:** `NONE`

---

## 16. LLO

**Description:**  Lyapunov only. Refers to the "last layer of one" technique. If true, the last layer of the neural network is initialized to all ones. Requires last layer to have a positive definite activation function (eg square).

**Default Value:** `False`

---

## 17. ROUNDING

**Description:** Rounding precision for symbolic calculation. Lower rounding can improve performance.

**Default Value:** `3`

---

## 18. N_VARS

**Description:** Number of variables.

**Default Value:** `Required`

---

## 19. N_HIDDEN_NEURONS

**Description:** List/Tuple specifying the number of hidden neurons in the certificate neural network.

**Default Value:** `[10,]`

---

## 20. ACTIVATION

**Description:** Activation type tuple. Defines the activation function for each hidden layer of the certificate neural network.

**Options:**

* IDENTITY: No transformation.
* LINEAR: Linear activation.
* SQUARE: Square of the input.
* POLY_2, ..., POLY_\<i\> , ...,  POLY_8: Polynomial activations of order 1 to $i$.
* EVEN_POLY_4, ..., EVEN_POLY_\<i\> , ..., EVEN_POLY_10: Even polynomial activations or order 2 to $i$.
* TANH, SIGMOID, SOFTPLUS, COSH: Neural activation functions, dReal only.
* RELU: Rectified Linear Unit (discrete time only).

**Default Value:** `[SQUARE,]`

---

## 23. CTRLAYER

**Description:**  A list of integers defining neurons in control network. Must be specified if dynamical model has control inputs. Set as

**Default Value:** `None`

---

## 24. CTRLACTIVATION

**Description:** Control activation tuple. Defines the activation function for each hidden layer of the control neural network.

**Default Value:** `None`

---

## 25. N_HIDDEN_NEURONS_ALT

**Description:** For certificates involving two neural network functions (SWA, RSWA), this parameter defines the number of hidden neurons in the second neural network.

**Default Value:** `(10,)`

---

## 26. ACTIVATION_ALT

**Description:** For certificates involving two neural network functions (SWA, RSWA), this parameter defines the activation function for each hidden layer of the second neural network.

**Default Value:** `(SQUARE,)`

---

## 27. SEED

**Description:** Set the seed for reproducibility. If not specified, no seed is set.

**Default Value:** `None`

---

 Adjust these settings as per your requirements to get the desired behavior from the program.
