.. _algo:

Algorithms
===========

Overview
++++++++++

This section holds my implementation of some basic, yet important machine learning algorithms. The algorithms are
implemented in an object-oriented manner so that they could potentially be extended or incorporated to :ref:`project`,
 :ref:`utility` or other algorithms. All implementation should have unittest and documentation.


Tree
++++++++++

Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning
algorithms, decision tree algorithm can be used for solving regression and classification problems too.

Decision Tree Algorithm Pseudocode

#. Place the best attribute of the dataset at the root of the tree.
#. Split the training set into subsets. Subsets should be made in such a way that each subset contains data with the
same value for an attribute.
#. Repeat step 1 and step 2 on each subset until you find leaf nodes in all the branches of the tree.



Attribute Selection
----------------------

The primary challenge in the decision tree implementation is to identify which attributes do we need to consider as the root node and each level. Handling this is know the attributes selection. We have different attributes selection measure to identify the attribute which can be considered as the root note at each level.
The popular attribute selection measures:
    * Information gain
    * Gini index

If dataset consists of “n” attributes then deciding which attribute to place at the root or at different levels of the
tree as internal nodes is a complicated step. By just randomly selecting any node to be the root can’t solve the issue.
If we follow a random approach, it may give us bad results with low accuracy.

For solving this attribute selection problem, researchers worked and devised some solutions. They suggested using some
criterion like information gain, gini index, etc. These criterions will calculate values for every attribute.
The values are sorted, and attributes are placed in the tree by following the order i.e, the attribute with a high
value(in case of information gain) is placed at the root.

While using information Gain as a criterion, we assume attributes to be categorical, and for gini index,
attributes are assumed to be continuous.


Reference:
    .. [Ref] http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/#comments
    .. [Ref] http://scikit-learn.org/stable/modules/tree.html

.. _information gain:

Information gain (Entropy)
--------------------------

InfoGain class: the class takes the following input for initialization

    + **train_X**: training data X, a Pandas dataframe
    + **train_y**: training data y, a Pandas series
    + **params**:
                Other parameters that defines the way of split the data. For example, it could be split_mode = "random",
                which means that, to split each column, a random threshold will be selected from the column; or
                split_threshold = {'A':1, 'B':2, 'C':3, ...}, which means that, 1 will be used as the threshold to split
                column 'A', 2 will be used to split column 'B', etc.

    + **example**:

            |    ``info_gain = InfoGain(train_X, train_y, split_mode="random")``
            |    ``print(info_gain.information_gain)``