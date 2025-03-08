# BITBLE
Binary Tree Block encoding for quantum circuits


## Install Python and Python pachages

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

Put the folder "BITBLE" under the root directory of your project

## mindquantum - implementation





## pyqpanda - implementation


### State Preparation ###

```
    from pyqpanda import *
    from BITBLE.qpanda import statepreparation

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
    from pyqpanda import *
    from BITBLE.qpanda import blockencoding
    import numpy as np

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



