# JFB: Jacobian-Free Backpropagation for Implicit Networks (2022)
Source: Jacobian-Free Backpropagation for Implicit Networks (JFB).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images (SVHN, MNIST, CIFAR-10) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Class labels (inferred) | 0D (inferred) | Not specified in the paper. |

## Summary
The paper evaluates JFB-trained implicit networks on image classification tasks using SVHN, MNIST, and CIFAR-10. These tasks operate over 2D image inputs and produce 0D class labels (inferred). The model uses an internal latent representation, indicating constructed state dynamics (inferred), while task dynamics and attention behavior are not specified.

## Evidence
### Task: Image classification
- "three benchmark image classification datasets licensed under CC-BY-SA: SVHN (Netzer et al. 2011), MNIST (LeCun, Cortes, and Burges 2010), and CIFAR-10 (Krizhevsky and Hinton 2009)." (Section Classification)
- "The implicit portion of the network uses a latent space  $\mathcal{U}$ , and data is mapped to this latent space by  $Q_{\Theta}: \mathcal{D} \to \mathcal{U}$ ." (Section Implicit Network Formulation)
- Inference: Input dimension is 2D (x, y) because the task is image classification; output is class labels with 0D outputs for classification; state is constructed due to the latent space used by the model.
