# BITBLE
Binary Tree Block encoding for quantum circuits (python version)

This algorithm is bulit on top of [mindquantum](https://www.mindspore.cn/mindquantum/docs/en/r0.6/index.html)/[PyQPanda](https://github.com/OriginQ/pyQPanda-Toturial/blob/master/source/index.rst) in Python

## Install Python and Python packages

1. Download and install [Anaconda](https://www.anaconda.com/download)

2. Open the Anaconda Prompt
   
3. Create a virtual environment with Python 3.9.11 as an example

   ```
   conda create -n quantum python=3.9.11 -y
   conda activate quantum
   ```

3. Install Python packages

   ```
   pip install numpy
   ```
   
   ```
   pip install mindquantum
   ```
   
   ```
   pip install pyqpanda
   ```

## Note

Put the folder "[bitble](https://github.com/ChunlinYangHEU/BITBLE_python/tree/main/bitble)" under the root directory of your project


## mindquantum - implementation

### State Preparation ###
```
import numpy as np
from mindquantum import *
from bitble.mindquant import statepreparation

if __name__ == '__main__':
    """
         Tests the preparation of a random complex(real)-valued quantum state using compressed state preparation.

        Args:
            n (int): The number of qubits used to represent the state. The state will have 2^n amplitudes.
            epsilon (float): The threshold for filtering out small rotation angles during the compressed
                state preparation process. This parameter controls the trade-off between circuit complexity
                and accuracy.

        Returns:
            None: The function prints the constructed quantum circuit, the prepared state, the original
                state, and the Frobenius norm of their difference.
    """
    n = 3
    epsilon = 0
    # state = np.random.randn(2 ** n)
    state = (np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n))
    state = state / np.linalg.norm(state)

    circuit = statepreparation.compressed_state_preparation(state, list(range(n)), epsilon=epsilon)
    print(circuit)

    unitary = circuit.matrix()
    res_state = statepreparation.get_prepared_state(unitary)

    if all(np.isreal(state)):
        print('--------Real state preparation--------')
    else:
        print('--------Complex state preparation--------')
    print('The prepared state (s):')
    print(res_state)
    print('')
    print('The state to be prepared (s0):')
    print(state)
    print('')
    print('||s - s0||_F:')
    print(np.linalg.norm(res_state - state))
```
Output:
```
      ┏━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━┓                                     
q0: ──┨ RZ(-1.1109) ┠─┨ RY(1.616) ┠───■─────────────────────■─────────↯─
      ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━┛   ┃                     ┃           
      ┏━━━━━━━━━━━━┓                ┏━┻━┓ ┏━━━━━━━━━━━━━┓ ┏━┻━┓         
q1: ──┨ RY(1.8916) ┠────────────────┨╺╋╸┠─┨ RY(-0.0533) ┠─┨╺╋╸┠───■───↯─
      ┗━━━━━━━━━━━━┛                ┗━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━┛   ┃     
      ┏━━━━━━━━━━━━┓                                            ┏━┻━┓   
q2: ──┨ RY(1.9899) ┠────────────────────────────────────────────┨╺╋╸┠─↯─
      ┗━━━━━━━━━━━━┛                                            ┗━━━┛   
                                                                        
q0: ───────────────────■──────────────────────────────────────────■───↯─
                       ┃                                          ┃     
                       ┃                                          ┃     
q1: ───────────────────╂────────────────────■─────────────────────╂───↯─
                       ┃                    ┃                     ┃     
      ┏━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━━━━┓ ┏━┻━┓   
q2: ──┨ RY(0.0337) ┠─┨╺╋╸┠─┨ RY(0.2178) ┠─┨╺╋╸┠─┨ RY(-0.2323) ┠─┨╺╋╸┠─↯─
      ┗━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━┛   
                                                                        
      ┏━━━━━━━━━━━━━┓                                                   
q0: ──┨ RZ(-0.2841) ┠───■────────────────────■────────────────────────↯─
      ┗━━━━━━━━━━━━━┛   ┃                    ┃                          
      ┏━━━━━━━━━━━━┓  ┏━┻━┓ ┏━━━━━━━━━━━━┓ ┏━┻━┓                        
q1: ──┨ RZ(0.5327) ┠──┨╺╋╸┠─┨ RZ(1.6212) ┠─┨╺╋╸┠───■──────────────────↯─
      ┗━━━━━━━━━━━━┛  ┗━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━┛   ┃                    
      ┏━━━━━━━━━━━━┓                             ┏━┻━┓ ┏━━━━━━━━━━━━┓   
q2: ──┨ RZ(2.0334) ┠─────────────────────────────┨╺╋╸┠─┨ RZ(-1.196) ┠─↯─
      ┗━━━━━━━━━━━━┛                             ┗━━━┛ ┗━━━━━━━━━━━━┛   
                                                                        
q0: ────■───────────────────────────────────────■─────                  
        ┃                                       ┃                       
        ┃                                       ┃                       
q1: ────╂────────────────────■──────────────────╂─────                  
        ┃                    ┃                  ┃                       
      ┏━┻━┓ ┏━━━━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓ ┏━┻━┓                     
q2: ──┨╺╋╸┠─┨ RZ(2.3318) ┠─┨╺╋╸┠─┨ RZ(0.31) ┠─┨╺╋╸┠───                  
      ┗━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛ ┗━━━┛                     

--------Complex state preparation--------
The prepared state (s):
[-0.11717851-0.1918681j   0.07393187+0.34585958j  0.15615516+0.36916417j
 -0.27142767+0.25972169j -0.06083434+0.20457388j  0.34623429+0.0192198j
 -0.19897308-0.08064977j -0.44480752+0.33590001j]

The state to be prepared (s0):
[-0.11717851-0.1918681j   0.07393187+0.34585958j  0.15615516+0.36916417j
 -0.27142767+0.25972169j -0.06083434+0.20457388j  0.34623429+0.0192198j
 -0.19897308-0.08064977j -0.44480752+0.33590001j]

||s - s0||_F:
3.176960279072491e-16
```
### Block Encoding ###

```
from mindquantum import *
from bitble.mindquant import blockencoding
import numpy as np


if __name__ == '__main__':
    n = 2
    epsilon = 0
    matrix = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)
    matrix = matrix / np.linalg.norm(matrix)
    """
        Tests the block encoding of a random complex-valued matrix using compressed quantum circuit.

        Args:
            n (int): The number of qubits used to represent the matrix. The matrix will have dimensions
                2^n × 2^n.
            epsilon (float): The threshold for filtering out small rotation angles during the compressed
                quantum circuit construction. This parameter controls the trade-off between circuit
                complexity and accuracy.

        Returns:
            None: The function prints the constructed quantum circuit, the encoded matrix, the original
                matrix, and the Frobenius norm of their difference.
        """


    qubits = list(range(2 * n))
    circuit = blockencoding.compress_qcircuit(matrix, qubits, epsilon=epsilon)
    print(circuit)

    unitary = circuit.matrix()
    res_matrix = blockencoding.get_encoded_matrix(unitary, n)

    print('The encoded matrix (A):')
    print(res_matrix)
    print('The matrix to be encoded (A0):')
    print(matrix)
    print('||A - A0||_F:')
    print(np.linalg.norm(res_matrix - matrix))
```
Output:
```
      ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓    
q0: ──┨ RZ(0.4574) ┠─┨╺╋╸┠─┨ RZ(-0.4706) ┠─┨╺╋╸┠─┨ RZ(0.2955) ┠─┨╺╋╸┠─↯─ 
      ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛    
                       ┃                     ┃                    ┃      
q1: ───────────────────╂─────────────────────╂────────────────────╂───↯─ 
                       ┃                     ┃                    ┃      
                       ┃                     ┃                    ┃      
q2: ───────────────────╂─────────────────────■────────────────────╂───↯─ 
                       ┃                                          ┃      
                       ┃                                          ┃      
q3: ───────────────────■──────────────────────────────────────────■───↯─ 
                                                                         
      ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓ ┏━━━┓   
q0: ──┨ RZ(-0.6962) ┠─┨╺╋╸┠─┨ RY(2.1127) ┠─┨╺╋╸┠─┨ RY(-0.1469) ┠─┨╺╋╸┠─↯─
      ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛ ┗━┳━┛   
                        ┃                    ┃                     ┃     
q1: ────────────────────╂────────────────────╂─────────────────────╂───↯─
                        ┃                    ┃                     ┃     
                        ┃                    ┃                     ┃     
q2: ────────────────────■────────────────────╂─────────────────────■───↯─
                                             ┃                           
                                             ┃                           
q3: ─────────────────────────────────────────■─────────────────────────↯─
                                                                         
      ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓                         
q0: ──┨ RY(-0.0678) ┠─┨╺╋╸┠─┨ RY(-0.142) ┠─┨╺╋╸┠──────────────────────↯─ 
      ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛                         
                        ┃                    ┃   ┏━━━━━━━━━━━━┓ ┏━━━┓    
q1: ────────────────────╂────────────────────╂───┨ RY(1.3631) ┠─┨╺╋╸┠─↯─ 
                        ┃                    ┃   ┗━━━━━━━━━━━━┛ ┗━┳━┛    
                        ┃                    ┃                    ┃      
q2: ────────────────────╂────────────────────■────────────────────╂───↯─ 
                        ┃                                         ┃      
                        ┃                                         ┃      
q3: ────────────────────■─────────────────────────────────────────■───↯─ 
                                                                         
q0: ───────────────────────────────────────────────────────────────■───↯─
                                                                   ┃     
      ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓ ┏━┻━┓   
q1: ──┨ RY(0.1846) ┠─┨╺╋╸┠─┨ RY(-0.3132) ┠─┨╺╋╸┠─┨ RY(-0.1245) ┠─┨╺╋╸┠─↯─
      ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛ ┗━━━┛   
                       ┃                     ┃                           
q2: ───────────────────■─────────────────────╂─────────────────────────↯─
                                             ┃                           
                                             ┃                           
q3: ─────────────────────────────────────────■─────────────────────────↯─
                                                                         
q0: ─────────────────────────────────────────────────────────────────↯─  
                                                                         
      ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓     
q1: ──┨ RY(0.3799) ┠─┨╺╋╸┠─┨ RY(0.0751) ┠─┨╺╋╸┠─┨ RY(0.1456) ┠─┨╺╋╸┠─↯─  
      ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛     
                       ┃                    ┃                    ┃       
q2: ───────────────────╂────────────────────■────────────────────╂───↯─  
                       ┃                                         ┃       
                       ┃                                         ┃       
q3: ───────────────────■─────────────────────────────────────────■───↯─  
                                                                         
                            ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓        
q0: ────────────────────■───┨ RZ(-1.2829) ┠─┨╺╋╸┠─┨ RZ(-1.4147) ┠─↯─     
                        ┃   ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛        
      ┏━━━━━━━━━━━━━┓ ┏━┻━┓                   ┃                          
q1: ──┨ RY(-0.1376) ┠─┨╺╋╸┠───────────────────╂───────────────────↯─     
      ┗━━━━━━━━━━━━━┛ ┗━━━┛                   ┃                          
                                              ┃                          
q2: ──────────────────────────────────────────╂───────────────────↯─     
                                              ┃                          
                                              ┃                          
q3: ──────────────────────────────────────────■───────────────────↯─     
                                                                         
      ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓ ┏━━━┓                   
q0: ──┨╺╋╸┠─┨ RZ(0.1441) ┠─┨╺╋╸┠─┨ RZ(-2.0382) ┠─┨╺╋╸┠─────────────────↯─
      ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛ ┗━┳━┛                   
        ┃                    ┃                     ┃   ┏━━━━━━━━━━━━━┓   
q1: ────╂────────────────────╂─────────────────────╂───┨ RZ(-1.8526) ┠─↯─
        ┃                    ┃                     ┃   ┗━━━━━━━━━━━━━┛   
        ┃                    ┃                     ┃                     
q2: ────■────────────────────╂─────────────────────■───────────────────↯─
                             ┃                                           
                             ┃                                           
q3: ─────────────────────────■─────────────────────────────────────────↯─
                                                                         
q0: ────────────────────────────────────────────────────────────────↯─   
                                                                         
      ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━┓      
q1: ──┨╺╋╸┠─┨ RZ(1.7738) ┠─┨╺╋╸┠─┨ RZ(0.7078) ┠─┨╺╋╸┠─┨ RZ(0.303) ┠─↯─   
      ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━┛      
        ┃                    ┃                    ┃                      
q2: ────╂────────────────────■────────────────────╂─────────────────↯─   
        ┃                                         ┃                      
        ┃                                         ┃                      
q3: ────■─────────────────────────────────────────■─────────────────↯─   
                                                                         
q0: ────■──────────────────────────────────────────────────────────────↯─
        ┃                                                                
      ┏━┻━┓ ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━━┓ ┏━━━┓ ┏━━━━━━━━━━━━┓   
q1: ──┨╺╋╸┠─┨ RZ(-1.5594) ┠─┨╺╋╸┠─┨ RZ(-0.3798) ┠─┨╺╋╸┠─┨ RZ(0.0028) ┠─↯─
      ┗━━━┛ ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━━┛ ┗━┳━┛ ┗━━━━━━━━━━━━┛   
                              ┃                     ┃                    
q2: ──────────────────────────╂─────────────────────■──────────────────↯─
                              ┃                                          
                              ┃                                          
q3: ──────────────────────────■────────────────────────────────────────↯─
                                                                         
q0: ─────────────────────────■───╳─────■────────────────────■───↯─       
                             ┃   ┃     ┃                    ┃            
      ┏━━━┓ ┏━━━━━━━━━━━━┓ ┏━┻━┓ ┃   ┏━┻━┓ ┏━━━━━━━━━━━━┓ ┏━┻━┓          
q1: ──┨╺╋╸┠─┨ RZ(1.3001) ┠─┨╺╋╸┠─┃─╳─┨╺╋╸┠─┨ RY(0.1774) ┠─┨╺╋╸┠─↯─       
      ┗━┳━┛ ┗━━━━━━━━━━━━┛ ┗━━━┛ ┃ ┃ ┗━━━┛ ┗━━━━━━━━━━━━┛ ┗━━━┛          
        ┃                        ┃ ┃                                     
q2: ────╂────────────────────────╳─┃────────────────────────────↯─       
        ┃                          ┃                                     
        ┃                          ┃                                     
q3: ────■──────────────────────────╳────────────────────────────↯─       
                                                                         
      ┏━━━━━━━━━━━━┓                                                     
q0: ──┨ RY(-1.349) ┠────                                                 
      ┗━━━━━━━━━━━━┛                                                     
      ┏━━━━━━━━━━━━━┓                                                    
q1: ──┨ RY(-1.5645) ┠───                                                 
      ┗━━━━━━━━━━━━━┛                                                    
                                                                         
q2: ────────────────────                                                 
                                                                         
                                                                         
q3: ────────────────────                                                 

The encoded matrix (A):
[[-1.91333632e-01+0.19179827j -1.75090829e-01+0.02248911j
   8.67192857e-02-0.10243516j -3.18008805e-02-0.18122005j]
 [-2.39502433e-01+0.12802397j  9.14206875e-02-0.11511729j
   3.67289370e-02+0.12464116j -2.97753425e-02-0.02274609j]
 [-4.22473818e-01-0.11624631j  1.63120914e-01+0.30592713j
   6.15539570e-05+0.12368836j -3.08109069e-01+0.02940703j]
 [ 3.86273271e-02-0.14173991j -2.75500846e-01-0.02030555j
  -3.10171694e-01-0.12400485j -2.75093487e-01-0.14656077j]]
The matrix to be encoded (A0):
[[-1.91333632e-01+0.19179827j -1.75090829e-01+0.02248911j
   8.67192857e-02-0.10243516j -3.18008805e-02-0.18122005j]
 [-2.39502433e-01+0.12802397j  9.14206875e-02-0.11511729j
   3.67289370e-02+0.12464116j -2.97753425e-02-0.02274609j]
 [-4.22473818e-01-0.11624631j  1.63120914e-01+0.30592713j
   6.15539570e-05+0.12368836j -3.08109069e-01+0.02940703j]
 [ 3.86273271e-02-0.14173991j -2.75500846e-01-0.02030555j
  -3.10171694e-01-0.12400485j -2.75093487e-01-0.14656077j]]
||A - A0||_F:
4.672551908896126e-16
```

## pyqpanda - implementation


### State Preparation ###

```
import numpy as np
from pyqpanda import *
from BITBLE.qpanda import statepreparation

if __name__ == '__main__':
    n = 3
    epsilon = 0
    state = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
    state = state / np.linalg.norm(state)

    circuit = statepreparation.compressed_state_preparation(state, list(range(n)), epsilon=epsilon)
    print(circuit)

    unitary = circuit.matrix()
    res_state = statepreparation.get_prepared_state(unitary)

    print('The prepared state (s):')
    print(res_state)
    print('')
    print('The state to be prepared (s0):')
    print(state)
    print('')
    print('||s - s0||_F:')
    print(np.linalg.norm(res_state - state))
```
Output:
```
--------Complex state preparation--------

          ┌──┐ ┌──┐                                             ┌──┐                                             
q_0:  |0>─┤RZ├ ┤RY├ ─■─ ──── ─■─ ─── ──── ─■─ ──── ─── ──── ─■─ ┤RZ├ ─■─ ──── ─■─ ─── ──── ─■─ ──── ─── ──── ─■─ 
          ├──┤ └──┘ ┌┴┐ ┌──┐ ┌┴┐           │           ┌──┐  │  └──┘ ┌┴┐ ┌──┐ ┌┴┐           │                 │  
q_1:  |0>─┤RY├ ──── ┤X├ ┤RY├ ┤X├ ─■─ ──── ─┼─ ──── ─■─ ┤RZ├ ─┼─ ──── ┤X├ ┤RZ├ ┤X├ ─■─ ──── ─┼─ ──── ─■─ ──── ─┼─ 
          ├──┤      └─┘ └──┘ └─┘ ┌┴┐ ┌──┐ ┌┴┐ ┌──┐ ┌┴┐ ├──┤ ┌┴┐ ┌──┐ └─┘ └──┘ └─┘ ┌┴┐ ┌──┐ ┌┴┐ ┌──┐ ┌┴┐ ┌──┐ ┌┴┐ 
q_2:  |0>─┤RY├ ──── ─── ──── ─── ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RZ├ ─── ──── ─── ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ 
          └──┘                   └─┘ └──┘ └─┘ └──┘ └─┘ └──┘ └─┘ └──┘              └─┘ └──┘ └─┘ └──┘ └─┘ └──┘ └─┘ 
 c :   / ═
          


The prepared state (s):
[-0.27644244+0.1610026j  -0.01573856-0.34984595j -0.44353131+0.19808173j
  0.06034424+0.23572727j -0.20606243-0.10696451j  0.49705746+0.21671312j
  0.05255242-0.03075348j  0.20549343+0.29322989j]

The state to be prepared (s0):
[-0.27644244+0.1610026j  -0.01573856-0.34984595j -0.44353131+0.19808173j
  0.06034424+0.23572727j -0.20606243-0.10696451j  0.49705746+0.21671312j
  0.05255242-0.03075348j  0.20549343+0.29322989j]

||s - s0||_F:
3.7789906771029953e-16
```

### Block Encoding ###

```
import numpy as np
from pyqpanda import *
from BITBLE.qpanda import blockencoding


    n = 3
    epsilon = 0
    matrix = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)
    matrix = matrix / np.linalg.norm(matrix)

    num_qubits = 2 * n
    init(QMachineType.CPU)
    qubits = qAlloc_many(num_qubits)
    cbits = cAlloc_many(num_qubits)
    circuit = blockencoding.compress_qcircuit(matrix, qubits, epsilon=epsilon)
    print(circuit)

    unitary = get_unitary(circuit)
    unitary = np.array(unitary).reshape(2 ** num_qubits, 2 ** num_qubits)
    res_matrix = blockencoding.get_encoded_matrix(unitary, n)

    print('The encoded matrix (A):')
    print(res_matrix)
    print('')
    print('The matrix to be encoded (A0):')
    print(matrix)
    print('')
    print('||A - A0||_F:')
    print(np.linalg.norm(res_matrix - matrix))

```
Output:
```
--------Complex matrix--------

          ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐                      >
q_0:  |0>─┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├─── ─── ──── ─── ──── >
          ├──┤ └┬┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └┬┘ ├─┬┘ └┬┼──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ >
q_1:  |0>─┤RY├ ─┼─ ──── ─┼─ ──── ─┼─ ──── ─┼─ ──── ─┼─ ──── ─┼─ ──── ─┼─ ┤X├─ ─┼┤RY├ ┤X├ ┤RY├ ┤X├ ┤RY├ >
          └──┘  │        │        │        │        │        │        │  └┬┘   │└──┘ └┬┘ └──┘ └┬┘ └──┘ >
q_2:  |0>───── ─┼─ ──── ─■─ ──── ─┼─ ──── ─■─ ──── ─┼─ ──── ─■─ ──── ─┼─ ─┼── ─■──── ─■─ ──── ─┼─ ──── >
                │                 │                 │                 │   │                    │       >
q_3:  |0>───── ─■─ ──── ─── ──── ─■─ ──── ─── ──── ─■─ ──── ─── ──── ─■─ ─■── ────── ─── ──── ─■─ ──── >
                                                                                                       >
 c :   / ═
          

                                                 ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐                 >
q_0:  |0>─■─ ──── ─── ──── ─── ──── ─── ──── ─■─ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├─── ─── ──── ─── >
         ┌┴┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌┴┐ ├──┤ └┬┘ └──┘ └┬┘ └──┘ └┬┘ ├─┬┘ └┬┼──┐ ┌─┐ ┌──┐ ┌─┐ >
q_1:  |0>┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RY├ ┤X├ ┤RZ├ ─┼─ ──── ─┼─ ──── ─┼─ ┤X├─ ─┼┤RZ├ ┤X├ ┤RZ├ ┤X├ >
         └─┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └─┘ └──┘  │        │        │  └┬┘   │└──┘ └┬┘ └──┘ └┬┘ >
q_2:  |0>─── ──── ─┼─ ──── ─■─ ──── ─┼─ ──── ─── ──── ─┼─ ──── ─■─ ──── ─┼─ ─┼── ─■──── ─■─ ──── ─┼─ >
                   │                 │                 │                 │   │                    │  >
q_3:  |0>─── ──── ─■─ ──── ─── ──── ─■─ ──── ─── ──── ─■─ ──── ─── ──── ─■─ ─■── ────── ─── ──── ─■─ >
                                                                                                     >
 c :   / 
         

                                                                          ┌──────┐ 
q_0:  |0>──── ─■─ ──── ─── ──── ─── ──── ─── ──── ─■─ X─ ─■─ ──────── ─■─ ┤RY.dag├ 
         ┌──┐ ┌┴┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌─┐ ┌──┐ ┌┴┐ │  ┌┴┐ ┌──────┐ ┌┴┐ ├──────┤ 
q_1:  |0>┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┤RZ├ ┤X├ ┼X ┤X├ ┤RY.dag├ ┤X├ ┤RY.dag├ 
         └──┘ └─┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └┬┘ └──┘ └─┘ ││ └─┘ └──────┘ └─┘ └──────┘ 
q_2:  |0>──── ─── ──── ─┼─ ──── ─■─ ──── ─┼─ ──── ─── X┼ ─── ──────── ─── ──────── 
                        │                 │            │                           
q_3:  |0>──── ─── ──── ─■─ ──── ─── ──── ─■─ ──── ─── ─X ─── ──────── ─── ──────── 
                                                                                   
 c :   / 
         


The encoded matrix (A):
[[ 0.0717925 -0.29259169j -0.15164446+0.14305347j -0.07978028+0.04929215j
  -0.06278604+0.0579586j ]
 [-0.09443328-0.02357717j -0.26869318+0.06971573j -0.29271985+0.07387712j
  -0.10569737-0.17675179j]
 [ 0.06225808-0.35331841j  0.20143416+0.1131481j  -0.13867982+0.00457545j
  -0.00672616+0.29886571j]
 [ 0.26427384+0.11476363j -0.04505902+0.10454406j  0.10836367-0.36276957j
   0.04147132+0.31281208j]]

The matrix to be encoded (A0):
[[ 0.0717925 -0.29259169j -0.15164446+0.14305347j -0.07978028+0.04929215j
  -0.06278604+0.0579586j ]
 [-0.09443328-0.02357717j -0.26869318+0.06971573j -0.29271985+0.07387712j
  -0.10569737-0.17675179j]
 [ 0.06225808-0.35331841j  0.20143416+0.1131481j  -0.13867982+0.00457545j
  -0.00672616+0.29886571j]
 [ 0.26427384+0.11476363j -0.04505902+0.10454406j  0.10836367-0.36276957j
   0.04147132+0.31281208j]]

||A - A0||_F:
4.637263367040817e-16
```

## Matlab implementation

More block-encoding circuits with different normalziation factor in low circuit size can be found [https://github.com/zexianLIPolyU/BITBLE-SIABLE_matlab](https://github.com/zexianLIPolyU/BITBLE-SIABLE_matlab) .


