.. _nice-dataset:

NICE dataset
##########
This is a toy dataset containing NICE[1, 4](N-body Iterative Contraction of Equivariants) features for first 500 configurations of the dataset[2, 3] with randomly displaced methane configurations. 

Function Call
-------------
.. function:: skmatter.datasets.load_nice_dataset

Data Set Characteristics
------------------------
:Number of Instances: 500
:Number of Features: 160
The representations were computed using the NICE package[4] using the following definition of the NICE calculator:

.. code-block:: python

    StandardSequence(
        [
            StandardBlock(
                ThresholdExpansioner(num_expand=150),
                CovariantsPurifierBoth(max_take=10),
                IndividualLambdaPCAsBoth(n_components=50),
                ThresholdExpansioner(num_expand=300, mode="invariants"),
                InvariantsPurifier(max_take=50),
                InvariantsPCA(n_components=30),
            ),
            StandardBlock(
                ThresholdExpansioner(num_expand=150),
                CovariantsPurifierBoth(max_take=10),
                IndividualLambdaPCAsBoth(n_components=50),
                ThresholdExpansioner(num_expand=300, mode="invariants"),
                InvariantsPurifier(max_take=50),
                InvariantsPCA(n_components=20),
            ),
            StandardBlock(
                None,
                None,
                None,
                ThresholdExpansioner(num_expand=300, mode="invariants"),
                InvariantsPurifier(max_take=50),
                InvariantsPCA(n_components=20),
            ),
        ],
        initial_scaler=InitialScaler(mode="signal integral", individually=True),
    )


References
----------
[1] Jigyasa Nigam, Sergey Pozdnyakov, and Michele Ceriotti. "Recursive evaluation and iterative contraction of N-body equivariant features." The Journal of Chemical Physics 153.12 (2020): 121101.

[2] Incompleteness of Atomic Structure Representations
Sergey N. Pozdnyakov, Michael J. Willatt, Albert P. Bartók, Christoph Ortner, Gábor Csányi, and Michele Ceriotti

[3] https://archive.materialscloud.org/record/2020.110

Reference Code
--------------
[4] https://github.com/lab-cosmo/nice
