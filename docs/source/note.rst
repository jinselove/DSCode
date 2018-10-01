Notes
*********


Algorithms
===========

Liear Regression
------------------

    #. `Linear, Ridge and Lasso Regression`_
        .. _Linear, Ridge and Lasso Regression: https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/
    #. Derive Univariate and Multivariate linear regression
        * `Derive univariate LR`_
            .. _Derive univariate LR: https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression
        * `Derive univariate LR using max likelihood method`_
            .. _Derive univariate LR using max likelihood method: https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf
        * `Derive multivariate LR`_
            .. _Derive multivariate LR: http://www.public.iastate.edu/~maitra/stat501/lectures/MultivariateRegression.pdf
    #. Code
        * `Regularized linear methods`_
            .. _Regularized linear methods: https://www.kaggle.com/apapiu/regularized-linear-models
        * `Self implementation of linear regression`_
            .. _Self implementation of linear regression: https://www.kaggle.com/mosa94/linear-regression-implementations

Logistic Regression
--------------------
    #. Derive Univariate and Multivariate logistic regression
        * `logistic regression`_
            .. _logistic regression: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf
        * `Derive univariate Logistic regression using max likelihood function`_
            .. _Derive univariate Logistic regression using max likelihood function: http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/
        * `Logistic regression for more than two classes`_
            .. _Logistic regression for more than two classes: https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
    #. Kernels
        * `build logistic regression step by step`_
            .. _build logistic regression step by step: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

Decision Tree
--------------

    #. How Decision tree works
        *  two ways to calculate information gain: entropy = :math:`-\sum{p_j log(p_j)}`, which measures the randomness of
           the target after split) and gini index :math:`1-\sum{p_j^2}`, which measures the impurity of the target
           after split). The smaller Entropy or Gini are, the better.

        *  At the root node of each level, 1. calculate the entropy of the target, the sum is over number of classes in
           target. 2. And then for each feature (categorical or splitted to categorical), calculate the probability of
           each target label, and then we can get the entropy of this feature. The different between the feature
           entropy (smaller) and the root entropy will be the entropy gain because randomness decreases. Select the
           feature with the largest information gain to split on.

        *  When splitting continuous variable, use a threshold to split or segment the data yourself first. The
           threshold (I guess) is chosen as the median of that feature.

        *  Forward pruning. The tree model can overfit if it's too deep. To avoid overfitting, we can do forward
           pruning, i.e., control early stopping, for example, in Sklearn you can set min sample split, min sample
           leaf, min impurity decrease, etc.

        *  Post pruning. Grow the tree to extreme, which means that all the final nodes are pure, and then start
           trimming the trees, for each trim, get the model performance using the validation dataset and if
           the performance increases, keep the trim, otherwise ignore the trim. Sklearn does not support post
           pruning for now.  If you want to implement it yourself, start from `here(prune decision trees)`_
            .. _here(prune decision trees): https://stackoverflow.com/questions/49428469/pruning-decision-trees

    #. Advantage
        *  simple to interpret; able-to-handle multi-output problem; cost efficient (log(N or datapoints))

    #. Disadvantage
        *  easily overfitting (to fix this, need to set the minimum sample leaf or max tree depth;
           no post pruning supported so far in sklearn);  can be unstable, i.e., little change in data could generate
           a totally different tree (to fix this, using ensemble method); decision at each node is local,
           cannot guarantee to find a globally optimal tree (can be fixed in an emsemble learner where the
           features and samples are randomly sampled with replacement); decision trees create biased trees if
           the target is imbalanced.

        * `advantages and disadvantages explained in sklearn`_
            .. _advantages and disadvantages explained in sklearn: http://scikit-learn.org/stable/modules/tree.html

    #. How decision tree works for regression
        * `Check this link`_
            .. _Check this link: http://chem-eng.utoronto.ca/~datamining/dmc/decision_tree_reg.htm
        *  For regression model, the information gain is calculated using the standard derivation.  

    #. Kernels
        * `Study of tree and forest algorithms`_
            .. _Study of tree and forest algorithms: https://www.kaggle.com/creepykoala/study-of-tree-and-forest-algorithms/notebook

