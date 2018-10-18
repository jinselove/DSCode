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
           In practice, one can do the post pruning by minimizing cost complexitity function.(Reference to **ESL Chapter 9.2.2**).
           The algorithm is described as follows:
            .. image:: attachment/tree_pruning.png

    #. How Decision tree split on categorical predictors **(Reference to ESL 9.2.4, This is something I didn't realize until**
       **I wrote this note)**

        In decision trees, when binary splitting a predictor having q possible unordered values, there are :math:`2^{q-1}-1` possible partitions of the q values into two groups, and
        the computations become prohibitive for large :math:`q`. However, it can be proved that, if the problem is a binary classification
        problem, the computation can be simplified. We order the :math:`q` categories in the predictor according to the proportion falling
        into outcome class 1. Then we split this predictor as if it were an ordered predictor. One can show that this is the best
        split for this predictor. However, for multi-class classification problem or regression problem, no such simplifications
        are possible. And one has to look for other ways to handle categorical variables **(Reference: ESL 9.2.4)**.



    #. Advantage
        * able-to-handle multi-output problem;
        * cost efficient (log(N or datapoints))
        * relatively easy to construct
        * they produce interpretable models (if the trees are small)
        * they naturally incorporate mixtures of numeric and categorical predictor variables and missing values
        * they are invariant under strictly monotone transformations of the individual predictors, i.e., scaling and/or
          more general transformations are not an issue, and they are immune to the effects of predictor outliers.
        * they perform internal feature selection as an integral part of the procedure. They are thereby resistant to the inclusion
          of many irrelevant predictor variables.
    #. Disadvantage
        * easily overfitting (to fix this, need to set the minimum sample leaf or max tree depth;
          no post pruning supported so far in sklearn);
        * can be unstable, i.e., little change in data could generate a totally different tree (to fix this, using
          ensemble method);
        * decision at each node is local, cannot guarantee to find a globally optimal tree (can be fixed in an emsemble
          learner where the features and samples are randomly sampled with replacement); decision trees create biased
          trees if the target is imbalanced.
        * High variance: decision trees usually have high variance, for example, **CART** (classification and regression
          trees.)
        * Accuracy is usually low.


        * `advantages and disadvantages explained in sklearn`_
            .. _advantages and disadvantages explained in sklearn: http://scikit-learn.org/stable/modules/tree.html

    #. How decision tree works for regression
        * `Check this link`_
            .. _Check this link: http://chem-eng.utoronto.ca/~datamining/dmc/decision_tree_reg.htm
        *  For regression model, the information gain is calculated using the standard derivation.  

    #. Kernels
        * `Study of tree and forest algorithms`_
            .. _Study of tree and forest algorithms: https://www.kaggle.com/creepykoala/study-of-tree-and-forest-algorithms/notebook
    #. Useful resource
        * ESL Chapter 9.2(Very good)

MARS: Multivariate Adaptive Regression splines
----------------------------------------------
    #. Reference: ESL Chapter 9.4
    #. Implementation: `py-earth`_
        .. _py-earth: https://contrib.scikit-learn.org/py-earth/content.html#


Ensemble methods
------------------

Overview
++++++++++

The purpose of ensemble methods is to combine the predictions of several base estimators built with a given learning
algorithm in order to improve generalizability / robustness over a single estimator

    * Read the details at `Sklearn ensemble methods documentation`_
        .. _Sklearn ensemble methods documentation: http://scikit-learn.org/stable/modules/ensemble.html
    * And at `ensemble learning in Machine learning`_
        .. _ensemble learning in Machine learning: https://towardsdatascience.com/ensemble-learning-in-machine-learning-getting-started-4ed85eb38e00
    * And at `Bias variace tradeoff and how boosting and bagging deal with them`_
        .. _Bias variace tradeoff and how boosting and bagging deal with them: http://www.cs.cornell.edu/courses/cs578/2005fa/CS578.bagging.boosting.lecture.pdf
    * And at `Ensemble learning to improve machine learning results`_
        .. _Ensemble learning to improve machine learning results: https://blog.statsbot.co/ensemble-learning-d1dcd548e936
    * And at `What is the difference between bagging and boosting`_
        .. _What is the difference between bagging and boosting: https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/
Different ensemble methods
++++++++++++++++++++++++++++

    * **Averaging methods**
        * Mechanism:  The driving principle is to build several estimators independently and then to average their
          predictions. On average, the combined estimator is usually better than any of the single base estimator
          because its variance is reduced. Averaging methods does not try to combine weak models, instead, the base
          models are usually very complex, for example, fully developed decision trees.
        * Examples:
                    * **Bagging methods** (`Bagging methods`_)
                    * **Random Forest** (`Random forest`_),
                    * **Extremely randomized trees** (`Extremely randomized trees`_)

    * **Boosting methods**
        * Mechanism: Base estimators are build sequentially and one tries to reduce the bias of the combined estimator.
          The motivation is to combine several weak models to produce a powerful ensemble
        * Examples:
                    * **Adaboost** (`Adaboost`_)
                    * **Gradient Tree Boosting** (`Gradient Tree Boosting`_)

    * **Stacking methods**
        * Mechanism: stacking is an ensemble learning technique that combines multiple classification or regression models
          via a meta-classifier or a meta-regressor. The base level models are trained based on a complete training set, then
          the meta-model is trained on the outputs of the base level model as features.
        * Examples:
                The base level models of stacking often consists of different learning algorithms and therefore stacking
                ensembles are often heterogeneous.

.. _Bagging methods:
Bagging methods
++++++++++++++++++

    #. How does bagging method work?

        Generally speaking, bagging methods take random samples (could be subsets of data points or subsets of features)
        from the original data and form **strong** base estimators for each of the sampled data, and then average the
        prediction results of each bases estimator. **Notice that, bagging does not necessarily have to be bagging of
        decision trees, but it often is used as bagging of decision trees.**

        * The random samples could be either random subsets of data points or random subset of features
            * Take random subset of the data points. When random subsets are taken without replacement, i.e., each
              subset cannot be used in multiple base estimators, the algorithm is called **Pasting**; when random subsets
              are taken with replacement, i.e., each subset can be used in multiple base estimators, the algorithm is
              call **Bagging**, which is in short for **Bootstrap aggregation**.
            * Take random subset of the features. When random subsets of the dataset are drawn as subsets of the
              features, the method is known is **Random Subspaces**. Take random subset of both the data points
              and the features. When base estimators are built on subsets of both samples and features,
              the method is known as **Random Patches**.

    #. Implementation in Sklearn (`bagging classifier`_, `bagging regressor`_)
        .. _bagging classifier: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        .. _bagging regressor: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html

        * Parameters:
            * Base_estimator: a classification or regression model, by default, it is a decision tree, but it can be
              anything else.
            * N_estimators: control number of base estimators, by_default=10
            * Max_samples or max_features: control the size of the size of the subsets in terms of samples and features,
              respectively.
            * Bootstrap: (by default = True) and bootstrap_features (by default  = False, and usually set to be False)
              control when the subsets are taken with replacement or not. Bootstrap == True usually performs better than
              False, I.e., Bagging performs better than Pasting.
            * Oob_score: control whether the generalization score can be calculated using out-of-bag sampels.
            * Warm_start: if true, reuse the solution of the previous call to fit and add more estimators to the ensemble.


        * Attributes:
            * base_estimator: unfitted base estimator
            * estimators: list of fitted base estimator (list of estimators)
            * Estimators_samples: the subset of drawn samples for each base estimator. (list of arrays)
            * Estimators_features: the subset of drawn features for each base estimator. (list of arrays)
            * Classes: the class labels (array of shape n_classes, for example, [0,1])
            * N_classes: the number of classes
            * Oob_score: score the training dataset obtained using out-of-bag estimate

    #. Pros and Cons:

        * Compared to decision tree
            * more robust and insensitive to the changes in data because averaging over multiple estimators
            * The variance is reduced by introducing randomness into its construction procedure and average the results
              the results from all estimators.

                .. image:: attachment/average_reduce_variance.png
                   :scale: 50 %

                .. image:: attachment/variance_of_bagging.png
                   :scale: 50 %
            * **Avoid overfitting** since each base estimator only use a subset of samples or features, thus could avoid
              fitting (overfitting usually happens when a estimator is fitted over the whole dataset)

        * Compared to boosting methods
            * Bagging has little effort on bias. Boosting can reduce bias by averaging
            * As bagging provides a way to reduce the risk of overfitting and the variance, it works best with strong
              and complex base estimators, for example, fully developed decision trees. While boosting methods usually
              work best with weak models, for example, shallow decision trees.



.. _Random forest:
Random Forest
++++++++++++++
    #. How does Random Forest work?

       Random forest is also a averaging ensemble method, it's like bagging of decision trees. But simple bagging of
       of decision trees have the problem that, the decision trees can have a lot of structural similarities and in turn
       have high correlation in their predictions even though each decision tree grows on a subset of the data. This high
       correlation could harm the prediciton ability of the ensemble method which works the best if the predictions
       from the sub-models are uncorrelated or at best weakly correlated. Random forest improves bagging of decision trees
       by guarantee that the predictions from all the the subtrees have less correlation. It is a simple tweak. In CART
       or bagging of CART, when selecting a split point, the learning algorithm is allowed to look through all variables
       and all variable values in order to select the most optimal split-point. The random forest algorithm changes
       this procedure so that the learning algorithm is limited to a random sample of features of which to search at each split.

       In other words, in Random forest:
        * Each tree is built using a bootstrap sample subset data points of the original data. (This can be turned on
          or turned off using "bootstrap" in sklearn. When bootstrap is off, each individual tree use all the samples)
        * Different from bagging of decision tree, random forest brings more randomness. In bagging of tree models,
          once each tree use a sample subset or a feature subset, and it does not change when growing the tree.
          However, in random forest, when growing each tree, when splitting the node, we don't select the feature that
          has the max information gain from all the features, but from a random subset of all the features. This random
          subset of features are different at each split.

    #. Pros (compared to CART and bagging of CART)
        * The variance is reduced because randomness in introduced and the results are averaged, so variance decreased.
        * The bias usually increases slightly with respect to a single decision tree, but the decrease in variance
          usually can compensate for the increase in bias, hence yielding an overall better model.

    #. Implementation in Sklearn (`Random Forest classifier`_, `Random Forest regressor`_)

        .. _Random Forest classifier: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        .. _Random Forest regressor: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        * In sklearn, the results is an average of the voting probability for each class in each tree, not the most
          voted class in each tree. This is different than the original paper.
        * Empirical good parameters to use
            * Max_features = n_features (for regression), max_features = sqrt(n_features) for classification. But in practice,
              it's recommended to do grid search over max_features as well.
            * Max_depth = None and min_sample_split=2(fully developed trees) which are both default values. But the
              problem is that it could consume a lot of memory to have fully developed trees. Thus practically,
              use grid search cv for min_sample_split for range(2,10,2) is a good idea, or search for max depth.
              And look at the grid search results for all parameter combination and select a simpler model if the
              performance is similar to the best but more complicated model.
            * When bootstrap is true, we can set oob_score = True so that the generalization accuracy can be estimated
              on the oob samples.(Notethat, in ExtraTrees in sklearn, boostrap by default is false, )

.. _Extremely randomized trees:
Extremely randomized trees
+++++++++++++++++++++++++++
    #. How does it work?
        Extremely randomized tree is different to Random Forest for the following reasons:
            * Each tree use all the data points instead of a bootstrap sample. (This can also be turned on or off in
              sklearn, by default bootstrap is false in Sklearn)
            * The split algorithm is different. At each node, similar to random forest, Extremely randomized trees
              also try to select a subset of features from all features and split these features and see which variable
              gives the most information gain. But the difference is that, extremely randomized trees split each variable
              totally randomly. For example, for feature A (no matter it is categorical or continuous), we first
              calculate the min and max of feature A, and then generate a split threshold from the uniform distribution
              between [A_min, A_max], and use this threshold to split feature A. Here is the split algorithm from the
              original paper.

              .. image:: attachment/extra_tree_split_algorithm.png
                :scale: 50 %
    #. Pros and Cons
        * Compared to Random forest
            * Extremely randomized tree model has even smaller variance but greater bias

.. _Adaboost:
Adaboost
+++++++++++

    #. How Adaboost classifier works? (`Sklearn Adaboost`_, ESL Chapter 10.)

    .. _Sklearn Adaboost: http://scikit-learn.org/stable/modules/ensemble.html#zzrh2009

    Adaboost.M1 is the most popular Adaboost algorithm, developed by Freund and Schapire in 1997. The basic idea of Adaboost
    is to build a series of weak estimators sequentially and finally average the predictions of each weak estimators by weights.
    The i-th estimator :math:`G_m` where m is from 1 to M, is built on the weighted data, :math:`\alpha_i X`. For the first
    estimator :math:`G_1(\alpha_1 X)`, :math:`\alpha_1 = \frac{1}{N}X`, i.e., the data are weighted using the same weight. And
    the estimator is equivalent to a estimator built on the original dataset. Then the estimator :math:`G_1` is
    reapplied to data :math:`X` (without weights) to make predictions, the data points that are miss classified are reweighted to highlight
    their importances, i.e., we get :math:`\alpha_2`, and then build the second estimator :math:`G_2(x)`.
    Keep repeating this process until M estimators and weights are formed. Finally, we get the weighted estimator
    :math:`G(X) = \text{sign}\sum_{m=1}^M\alpha_m G_m(X)`

    .. image:: attachment/adaboost_workflow.png
    .. image:: attachment/adaboost_algorithm.png

    **Note:**
      Adaboost is equivalent to **Stagewise Additive modeling** using **Exponential loss function**, i.e., :math:`L(y_i, f(x_i)) = Exp(-y_i f(x_i))`.
      In training data set, we could see that the misclassification error reduce to zero earlier than exponential loss
      as Boosting iteration continues (more and more base estimators are added to the model). For example, after M = 250,
      the misclassification error in the training dataset is already zero, but the exponential error still keep decreasing
      as M increases. It might seem that M = 250 is good enough, however, when we apply the model the test data, we will
      see that the misclassification error in the test data keeps increasing after M = 250, i.e., the model keeps improving
      after M = 250. This show that Adaboost is not optimizing training-set misclassification error, instead, it is optimizing
      the exponential loss, which is more sensitive to changes in the estimated class probabilities. **(Reference to ESL 10.4, 10.5)**

    .. image:: attachment/exponential_loss.png

    #. How Adaboost regressor works? (`Adaboost R2`_)
    .. _Adaboost R2: https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf

    Adaboost.R2 is the most popular Adaboost algorithm for regression, which is implemented in Sklearn. The difference with Adaboost
    classifier is the way of calculating the weight. In short, the weight applied on data :math:`(x_i,y_i)` in the next estimator
    depends on the loss of prediction :math:`y_i^p`, for example, if the linear loss function is :math:`L_i=\frac{|y^p(x_i)-y_i|}{D}`, where
    :math:`D=sup{|y_i^p(x_i)-y_i|}` for i = 1,...,N. Using this loss for data :math:`(x_i,y_i)`, we can build the weight of it. The
    higher the loss, the larger the weight is.


.. _Gradient Tree Boosting:
Gradient Tree Boosting
++++++++++++++++++++++++
    Of all the well-known learning methods, decision trees come closest to meeting the requirements for serving as an
    off-the-shelf procedure for data mining. They have many good properties, for example, 1) relatively easy to construct
    and they produce interpretable models (if the trees are small) 2) they naturally incorporate mixtures of numeric and categorical
    predictor variables and missing values 3) they are invariant under strictly monotone transformations of the individual predictors,
    i.e., scaling and/or more general transformations are not an issue, and they are immune to the effects of predictor outliers.
    4) they perform internal feature selection as an integral part of the procedure. They are thereby resistant to the inclusion
    of many irrelevant predictor variables. However, trees have one aspect that prevent them from being the ideal tool for
    predictive modeling, namely accuracy. They seldom provide predictive accuracy comparable to the best that can be achieved
    with the data at hand.

    Boosting decision trees can improve their accuracy, often dramatically. However, some advantages for trees
    that are sacrificed by boosting are **speed, interpretability**, and, for AdaBoost, robustness against overlapping class distributions
    and especially mislabeling of the training data.

    A **Gradient boosted model (GBM)** is a generalization of tree boosting that attempts to mitigate these problems, so
    as to produce an accurate and effective off-the-shelf procedure for data mining.

    #. How does it work?
        * `How to explain gradient boosting`_ (amazing)
        * `kaggle master explains gradient boosting`_
        * ESL Chapter 10

    .. _How to explain gradient boosting: http://explained.ai/gradient-boosting/index.html
    .. _kaggle master explains gradient boosting: http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/


Stacking
+++++++++++++++++++++
    #. How does it work?
        * The algorithm below summarizes stacking
        .. image:: attachment/stacking_algorithm.png



Data preprocessing
==================

Process categorical variables
-----------------------------
    Some algorithms can handle categorical variables naturally, for example, tree based models. However, in ML practice, it
    is more usual to handle categorical variables in data processing stage and before feeding to the algorithms. The question
    is, what is the best way to handle categorical variable? The options are as follows:

    * **Categorical Encoding (Leave them alone)** :
        This only works for algorithms that can deal with categorical variable naturally, for example, decision trees.
        However, it has been shown that this is the best way to handle categorical variables, no matter how many categories are
        in the feature (`Visiting Categorical Features and Encoding in Decision Trees`_). In decision trees, when binary
        splitting a predictor having q possible unordered values, there are :math:`2^{q-1}-1` possible partitions of the q values into two groups, and
        the computations become prohibitive for large :math:`q`. However, it can be proved that, if the problem is a binary classification
        problem, the computation can be simplified. We order the :math:`q` categories in the predictor according to the proportion falling
        into outcome class 1. Then we split this predictor as if it were an ordered predictor. One can show that this is the best
        split for this predictor. However, for multi-class classification problem or regression problem, no such simplifications
        are possible. And one has to look for other ways to handle categorical variables **(Reference: ESL 9.2.4)**.

    * **Numeric Encoding**:

        Convert :math:`q` categories into numeric values from :math:`0` to :math:`q-1`. This usually works better than
        Other encoding methods if :math:`q<1000` (`Visiting Categorical Features and Encoding in Decision Trees`_:
        not sure if this is 100% correctly, but it is convincing to some extent). In Sklearn,  can use `LabelEncoder`_

    .. _LabelEncoder: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

    * **Dummy variables (One-Hot encoder)**
        Create :math:`q` new columns for the feature, and the values are binary. This usually performs worse than **Numeric Encoding**,
        and not recommended to use. In sklearn, one can use `Pandas get_dummies`_

    .. _Pandas get_dummies: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html

    * **Binary encoder**
        The objective of Binary encoding is to use binary encoding to hash the cardinalities into binary values. It stores
        the same information as One-Hot encoding using hash table which generate much less features. It could outperform
        **Numerical Encoding** when :math:`q` is large.

    Read `Visiting Categorical Features and Encoding in Decision Trees`_ for a detailed investigation/benchmark for all
    these encoding methods to handle categorical variables.

    .. _Visiting Categorical Features and Encoding in Decision Trees: https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931

Handle Missing values
---------------------

Data Transformation
--------------------

