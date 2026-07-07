# Machine Learning SS2026 - Course Summary

Prof. Markus Mayer, TH Deggendorf. Condensed from the full study guides in `study_guides/`.
One compact reference covering everything taught across all 10 lectures plus the project and
presentation rules. Use it as a quick-inject context file for study, project work, and exam prep.

Primary sources the course is built on:
- ITSL / ISLR = An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani, Taylor), free at statlearning.com. The course backbone.
- MIT Open Learning Library, Introduction to Machine Learning (MIT-ITML).
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman, 2009) - harder, optional.

## Contents
1. Lecture 01: Introduction
2. Lecture 02: Machine Learning Overview
3. Lecture 03a: First Classifiers
4. Lecture 03b: ML Experiments and Evaluation
5. Lecture 04: Regression
6. Lecture 05: Logistic Regression and Bayesian Classifiers
7. Lecture 06: Feature Engineering
8. Lecture 07: Outliers, Cross-Validation and Resampling
9. Lecture 08: Optimization
10. Lecture 09: Advanced Classifiers (SVM, Decision Trees, Random Forests)
11. Lecture 10: Wrap-Up - An ML Project Guideline
12. Course Organization: Project and Presentation Rules

---

## Lecture 01: Introduction

### Course logistics
- ML course, TH Deggendorf, SS2026, Prof. Markus Mayer. 5 ECTS = 150h total: 60h presence (15 x 4h) + 90h self-study.
- No exam; assessment is project work. ~13 lectures, 2 mandatory project sessions, voluntary programming tutorial.
- One mandatory dataset check-in mid-semester. Presentations 8-10 July 2026 (may change). Professor does not program in lectures; coding is self-taught.
- Slides are "only a guideline"; real knowledge is in the primary sources.

### Primary sources
- ITSL / ISLR = An Introduction to Statistical Learning (James, Witten, Hastie, Tibshirani, Taylor; 2nd ed 2021, Python ed 2023). Free PDF (statlearning.com). Prof's #1 textbook; course backbone.
- MIT Open Learning Library, Introduction to Machine Learning (MIT-ITML). Free online, #1 online course. Python examples.
- Hardcore extra: The Elements of Statistical Learning (Hastie, Tibshirani, Friedman, 2009) - more math.
- Sources use different orderings/naming; same concept may have several names.

### What is AI (no single definition)
- There is NO fixed definition of AI; even two DIT professors differ, all valid. Two definitions given: Mayer's and the EU's.
- Mayer's view: "AI is a research field that tries to find algorithms/methods applicable to and solving many different tasks."
  - Most AI tasks are/were done by humans.
  - Algorithms made task-specific by deriving parameters from human knowledge OR from data.
  - Focus and methods of AI are ever-changing; solved problems stop being called AI ("AI effect"; Mayer's 2019 "I'm an AI expert" anecdote).
  - Defines AI as a field (not a machine property); deliberately avoids the words "artificial" and "intelligence"; engineer/task-focused. Calls it "a view, not a formal definition."
- Pattern recognition (Mayer's PhD field, glaucoma from OCT): "the research field of creating meaning from data (images, signals, text)"; uses ML, heuristics, optimization.
- EU definition: April 2019 High-Level Expert Group on AI. Focus is one diagram (map of subfields). Key claims: ML and reasoning each contain many techniques; robotics includes techniques OUTSIDE AI; all of AI falls within computer science.

### Two pillars of AI
- Split follows "where parameters come from":
  - Reasoning / Knowledge Representation = humans encode the rules (logic, Prolog, expert systems, ontologies, search). Transparent, needs no data, but does not scale; brittle on messy inputs.
  - Machine Learning = machine derives rules from data. Scales with data/compute; handles high-dimensional fuzzy inputs; needs data, can be opaque, can absorb bias.
- ML dominates: "Almost every current development in the AI field uses Machine Learning." Historical pivot: abundant data + compute made learn-from-data beat encode-by-hand.

### The three ML types (most exam-relevant trichotomy)
- Supervised: inputs + correct labels/targets; learn input-to-output mapping.
  - Regression: predict a continuous number.
  - Classification: predict a discrete category/label. (Most course algorithms are classifiers.)
- Unsupervised: inputs only, no labels; find hidden structure (clustering, e.g. k-means).
- Reinforcement: environment + reward signal; learn a policy of actions maximizing reward. OMITTED in this course (time).
- Deep Learning = a technique (neural nets) appearing across regression, classification, and RL - not a learning type.

### Course topics and goals
- Topics: what is ML; one classification example end-to-end (ML experiment, error measures, train vs test, kNN); Bayes classifiers (LDA, naive Bayes, QDA); linear (and logistic) regression; ML toolset (feature engineering/lifting/selection, outliers, cross-validation, resampling); intro to optimization (gradient descent, black-box); SVMs; decision trees; unsupervised (clustering); project guideline. RL omitted.
- NON-goals: not implement all algorithms (only basic ones: LDA, naive Bayes, linear regression, k-means, kNN, decision trees); not prove all math linkages (know where to find them).
- Goals: understand what ML can be; apply basic clustering/regression/classification; understand the mathematical difference between classifiers; construct a basic ML evaluation and know its pitfalls.
- scikit-learn algorithm map tells you WHICH method but omits the WHY. Goal = be a "thinking AI engineer" who justifies WHY. Depth-of-reasoning over breadth-of-recall.
- ITSL's four points: understand ideas behind techniques (how/when to use); learn simpler methods first; assess performance accurately (simpler methods often as good as fancier ones); ML is exciting/applied.
- Ethos: "Think first, prompt afterwards" (Prof. Mahadevan) - reason before using tools.

## Lecture 02: Machine Learning Overview

### What is ML (definitions)
- Prof's favorite (MIT-ITML): "Making decisions or predictions based on data" (short, operational).
- Wikipedia: methods that "learn" - leverage data to improve performance on tasks. ITSL: "tools for making sense of complex datasets." IBM: use data/algorithms to imitate human learning.
- Every definition contains "data" + "improving/predicting." Test a new problem: is there data? Is there something to predict/decide?
- ITSL vocabulary: calls ML "statistical learning"; a feature/predictor is a "predictor." Course follows ITSL notation but standard ML names. statistical learning ~= ML; predictor ~= feature ~= input variable.
- ML is not push-button: humans frame the problem, acquire/organize data, design solution space, choose algorithm+parameters, validate.

### Two foundations of ML (exam question)
- Statistics: main book is "Intro to Statistical Learning." Some classifiers built directly on statistical models (Bayes: naive Bayes, LDA, QDA). Key definition: "a statistic is anything that can be computed from the collected data."
- Optimization: without modern optimization the newest classifiers (SVMs, neural networks) cannot be trained. NN frameworks (TensorFlow, PyTorch) are actually general optimization frameworks.
- Together: training a model = using optimization to minimize a statistically motivated error measure on data.

### Three problem classes
- Four MIT-ITML design questions: (1) problem class, (2) assumptions, (3) evaluation criteria, (4) model type/algorithm.
- Unsupervised: no labels; find patterns. Subcategories: clustering, density estimation, dimensionality reduction. (One example only in course.)
- Supervised: inputs + specific outputs; learn a mapping. Subcategories: classification, regression. Main focus.
- Reinforcement: agent interacts with environment; policy state-to-action-to-reward. Not covered (historically few uses beyond game AI).
- Other names (sequence, active, transfer, semi-supervised) all reduce to these three.

### Notation (memorize)
- x_ij = single data value, sample i = 1..n, feature j = 1..p.
- n = number of observations/samples (rows). p = number of features per sample (columns).
- X = data matrix, size n x p. x_i = one sample (row of X, a p-vector).
- y_i = true target (a.k.a. expected / ground truth / gold standard), supervised only. y = column vector of targets.
- y_hat_i = model prediction = f(x_i).
- Most common error: confusing n (rows=samples) and p (columns=features). x_ij and y_i can be any type until a method needs numbers.

### Formal supervised-learning setup
- Given training set {(x_i, y_i)} for i = 1..n, find hypothesis f from hypothesis space F so predictions y_hat_i = f(x_i) minimize error E:
  - f_star = argmin over f in F of E(y, y_hat), with y_hat_i = f(x_i).
- f must generalize to NEW x, not memorize training pairs (previews overfitting).
- Learning = search F (optimization) for the f whose predictions best match labels (judged by statistics).

### Classification vs regression
- Split by domain of output y.
- Classification (qualitative): y_i in a finite set C = {c_1,...,c_K}, no numeric meaning, no arithmetic. Classifiers: kNN, Bayes (naive Bayes, LDA, QDA), decision trees/random forests, SVM, neural nets.
- Regression (quantitative): y_i in R, continuous; magnitude/order meaningful. Regressors: linear regression, logistic regression, kNN, neural nets.
- Rule of thumb: can you meaningfully average two valid outputs? Yes -> regression; no -> classification.
- Nuance (examined): logistic regression is usually used for classification (outputs probability, thresholded); classification can be argmax/one-hot over regression outputs; kNN and NNs do both. Numeric-looking labels (MNIST digits 0-9) are still classes.

### Error measure - classification (0-1 / misclassification)
- E = (1/n) * sum_i indicator(y_i != y_hat_i).
- indicator = 1 if wrong, 0 if correct. E = fraction misclassified. accuracy = 1 - E.
- Each error costs 1 (hence "0-1"). Alternatives: asymmetric / cost-sensitive loss (e.g. false negative >> false positive).
- Pitfall: class imbalance - 99% "not spam" scores 99% accuracy catching zero spam. Training-set error is optimistic.

### Error measure - regression (L2 / MSE)
- MSE: E = (1/n) * sum_i (y_i - y_hat_i)^2.
- residual = y_i - y_hat_i; squared residual >= 0. Squaring: removes sign cancellation, penalizes large errors more (sensitive to outliers), is smooth/differentiable (optimization-friendly, closed form for linear models).
- RMSE = sqrt(MSE) restores original units (human-readable). MSE units are the square of target units.
- Alternative: L1 / MAE = (1/n) * sum_i |y_i - y_hat_i| - more robust to outliers, non-smooth at 0.

### Minimizing error / "best"
- Notions: minimize expected loss (usual), minimize maximum loss (minimax/worst-case), characterize asymptotic behavior (infinite data limit).
- Goal is low error on UNSEEN data, not training data. Open questions (cliff-hanger): what data to train on, what "training" means, how to prove best -> lead to train/val/test splits, cross-validation, bias-variance, model selection.
- Course scope: mostly supervised; one unsupervised example; no RL. Reading: MIT-ITML Week 1, ITSL Ch. 2.1.

## Lecture 03a: First Classifiers

### The ML project
- Project = a temporary undertaking to produce a unique product/service/result. Three components:
  - Specific scope (result): an ML system for a task.
  - Schedule (start and end): deadlines.
  - Required resources (ML): computational power, data, recording equipment, people.
- Data and recording equipment are resources (constraints), not free givens. A system is only "good given these resources and this deadline."

### Steps of an ML project
1. Data acquisition (out of scope). 2. Data exploration. 3. Data preparation. 4. ML algorithm and parameter exploration. 5. Formal ML evaluation (the course goal). 6. Solution presentation. 7. Production/launch/monitoring/maintenance (out of scope).
- Order is forced: cannot prepare unexplored data, cannot choose algorithms before knowing data, cannot formally evaluate before exploring models. Course focus = steps 2-6.

### Exploration vs evaluation
- Both try features/models/parameters. Evaluation differs: thought-out order of experiments; deliberate choice of what to optimize; models/params that failed in exploration are OMITTED; often must repeat experiments on a common comparable data/parameter base (apples-to-apples).

### Experiment vs evaluation; hyperparameters
- One experiment = a single step in an evaluation using ONE fixed hyperparameter set; yields one quality number.
- An evaluation is made up of many experiments (structured comparison to find/justify the best config).
- Pipeline (supervised): Training data -> preprocessing -> feature extraction -> feature selection -> training -> classifier. Validation side: validation data -> preprocessing -> SELECTED feature extraction -> classifier -> quality measure. Never re-select features on validation data (information leak).
- Hyperparameter = a parameter influencing the training process/result but NOT a parameter of the classifier. Only classifier parameters compute predictions. kNN: k is a hyperparameter, kNN has no learned parameters. NN: learning rate/#layers = hyperparameters, weights = parameters.

### What an evaluation optimizes
- Three things: (1) features computed from data, (2) the ML model (family), (3) parameters and hyperparameters. Done jointly, in a principled/ordered way on a comparable basis; gives a guiding "red line" for the presentation.

### Properties of good data
- Suitable storage format (for size/type); ordered (date, class, patient ID); labeled and thoroughly checked by others (best: multiple observers); complete (all data needed present).
- "Not everything handed to you is good data." Garbage in -> garbage out; data quality sets the ceiling, not the algorithm.
- Explore BEFORE classifying: histograms, boxplots (know every element), scatter/line plots, summary statistics (mean, median, min, max). Scatter plot of two features with class markers is often clearest. Look at example data, especially corner cases. Understand every plot in detail.

### Classifier categorizations (name at least 3)
1. Parameterized vs parameter-free (does it compute parameters from data? kNN just stores data).
2. Bayesian (statistics-derived) vs the rest (human-reasoning/heuristic).
3. Linear vs non-linear (is the decision boundary a hyperplane or curved).
- Placements: kNN = parameter-free, human-reasoning, non-linear. Naive Bayes/LDA/QDA = Bayesian, parameterized; LDA linear, QDA non-linear.

### Bayes classifier (optimal)
- Assigns each observation to the most likely class; optimal for 0-1 error.
- B(x) = argmax over y_k of P(y_k | x) = argmax of [P(x|y_k) P(y_k) / P(x)].
- Terms: P(y_k|x) = a posteriori (maximize this); P(x|y_k) = likelihood; P(y_k) = a priori (base rate); P(x) = evidence.
- P(x) is the same for every class, so it DROPS OUT of the argmax: B(x) = argmax P(x|y_k) P(y_k).
- Optimal because picking max-posterior everywhere minimizes expected 0-1 loss (= misclassification probability). The unavoidable leftover from class overlap = Bayes error.
- Worked ex: priors P(spam)=0.4, P(ham)=0.6; likelihoods P(free|spam)=0.8, P(free|ham)=0.1. Scores: spam 0.32, ham 0.06 -> predict spam. Normalized posterior P(spam|free)=0.32/0.38=0.842. Decision unchanged by dividing by P(x).

### Why we still need other classifiers
- In the real world we do NOT know P(y_k) and P(x|y_k). Bayes classifier only applies exactly to simulated data with defined distributions.
- Approximate P(y_k) by counting class frequencies; for P(x|y_k) we must assume a form (e.g. normal) - assumptions may be false.
- Bayes-derived real classifiers: Naive Bayes, LDA, QDA, mixture-of-Gaussians (most assume normal features). Distribution-free: kNN.

### Decision boundary
- Informal: location in feature space where the classifier cannot decide between two classes (surface separating class regions).
- Formal (Bayes, classes y_1, y_2): { x | P(x|y_1) P(y_1) = P(x|y_2) P(y_2) }.
- For 2 classes, on the boundary each posterior = 0.5 (maximal indecision).
- Boundary shape = the linear/non-linear categorization; kNN gives a piecewise-linear, jagged boundary.

### kNN classifier
- Algorithm to classify new x: (1) compute Euclidean (L2) distance to ALL training vectors; (2) find the k nearest; (3) assign the majority class among them.
- Euclidean distance: d(x, x') = sqrt( sum over j of (x_j - x'_j)^2 ).
- Training = just storing the data (parameter-free, LAZY learner). No parameters computed; all work at prediction time. Training instant/cheap; prediction slow and memory-heavy.
- Justification: human reasoning ("things near each other are alike"), and it works well.
- Hyperparameters: (1) k = number of voting neighbors; (2) the distance metric (Euclidean/L2 default, Manhattan/L1 alternative).
- Effect of k: small k (e.g. 1) = flexible, jagged boundary, low bias/high variance, overfits/noise-sensitive; large k = smoother, robust, can underfit/blur small classes. Choosing k is a bias-variance trade-off (tune on validation).
- Ties: prefer odd k for 2 classes; or break by nearest of tied classes, weight by inverse distance (weighted kNN), or random.
- Pitfalls: MUST scale/normalize features (else large-magnitude features dominate distance); curse of dimensionality (all points nearly equidistant); slow prediction on large datasets.
- Worked (ISLR 2.4 #7): test point origin; distances Obs5=sqrt2~1.41(Green), Obs6=sqrt3~1.73(Red), Obs2=2(Red), Obs4=sqrt5~2.24(Green), Obs1=3(Red), Obs3=sqrt10~3.16(Red). K=1 -> Green (Obs5). K=3 -> Red (2:1). Highly non-linear Bayes boundary -> best K is small (flexible).

## Lecture 03b: ML Experiments and Evaluation

### Big picture
- ML experiment goal: honest estimate of generalization (performance on unseen data).
- Three pillars: (1) train/validation/test splitting; (2) features + normalization; (3) classification performance measures.
- Same preprocessing/feature pipeline applied to test data, but every pipeline parameter (norm mean/std, selected features) learned from TRAINING data ONLY. This prevents most data leakage.

### Train vs test; overfitting
- "A good/perfect result on training data is no guarantee and not even an indicator of good test performance." A perfect training result is itself an overfitting alarm.
- Overfitting = very flexible decision boundary fits training data well but does NOT generalize (high variance). Underfitting = too rigid, high train+test error (high bias).
- kNN illustration: k=1 -> train error 0, maximally overfit; larger k smooths boundary. Bayes error rate ~0.1304 (optimal, computable only in simulation); kNN k=10 ~0.1363. Tune k on validation, not test.
- Flexibility hierarchy (least->most flexible): linear (LDA, naive Bayes, linear SVM); polynomial (QDA, poly SVM, large-k kNN); NN/RBF-SVM/Random Forests; unrestricted trees, small-k kNN. "More flexible tends to generalize worse."

### Splitting rule
- Inviolable: train and test must have NO (unnecessary) connection (else data leakage).
- Split by the unit you want to generalize to: by patient (both-eyes OCT), by machine and/or time (predictive maintenance), by person (voice/emotion). Never naive random shuffle when samples correlated.
- Sizing: train big enough if increasing it stops changing results; test big enough if random test selections stop changing results. Common ~20% random test when samples independent. Validate split via multiple random splits + standard deviation of results.
- Three subsets: train = learn (parameters); validation = tune (hyperparameters); test = judge ONCE. Validation is effectively part of training. Test must never do validation's job. If short on data, fold validation into train (cross-validation), keep test sacred.

### Parameters vs hyperparameters
- Parameter = learned from data during training, needed for prediction, stored with model (NN weights, regression coefficients, class means).
- Hyperparameter = set before training (human/optimization), not learned (k in kNN, learning rate, max_depth, regularization). Subtypes: architecture, optimization, regularization.

### Features and normalization
- Feature = any number computable from the data. Classifier sees features, not raw data. Examples: mean/min/max/histogram bins (time series/images), word counts (text).
- More features NOT better: curse of dimensionality -> use feature selection.
- Why normalize: large-std feature dominates L2/Euclidean distance (kNN, k-means, SVM); numerical stability.
- z-score standardization (feature j over p samples): m_j = (1/p)sum x_ij; s_j = sqrt((1/p)sum(x_ij - m_j)^2); x_norm = (x_ij - m_j)/s_j. Result: mean 0, std 1.
- TRAP: compute m_j, s_j on TRAINING data alone, then apply to train, validation AND test. Fitting scaler on full/test data leaks test distribution (optimistic bias). sklearn: scaler.fit(X_train), then transform train and test; never fit on X_test or X_full.

### Classification metrics (the core)
- Error rate = (1/n)sum I(y_i != yhat_i); Accuracy = 1 - ErrRate = (TP+TN)/n.
- Accuracy FAILS under class imbalance: 100 diseased + 9900 normal, predict all normal -> 99% accuracy, zero patients found. Accuracy valid ONLY when classes balanced AND all error costs equal.
- Confusion matrix (rows=prediction, cols=truth): TP, TN, FP (false alarm = Type I), FN (miss = Type II). P=TP+FN, N=TN+FP.
- Sensitivity = Recall = TPR = TP/(TP+FN) = TP/P.
- Specificity = TNR = TN/(TN+FP) = TN/N. FPR = FP/N = 1 - Specificity (ROC x-axis).
- Precision = PPV = TP/(TP+FP). NPV = TN/(TN+FN).
- F1 = 2*prec*recall/(prec+recall) = 2TP/(2TP+FP+FN); harmonic mean; best 1, worst 0.
- Balanced accuracy = classwise averaged rate = average recall; for 2 classes (Sens+Spec)/2. Survives imbalance.
- Worked matrix (TP=70,FP=50,TN=850,FN=30,n=1000): Acc=0.92, Sens=0.70, Spec=0.944, Prec=0.583, NPV=0.966, F1=0.636, balanced acc=0.822.

### F1 criticisms (give >=2)
1. Ignores TN (not in formula). 2. Not symmetric / arbitrary choice of positive class. 3. Equal weight to precision/recall. 4. Depends on class ratio -> misleading for imbalanced classes.

### ROC and auROC
- For threshold-tunable classifiers (probability/score output). Plot TPR (y) vs FPR = 1-Specificity (x) by sweeping threshold.
- Curve hugging top-left = excellent; diagonal = random guessing.
- auROC: 1.0 = perfect separation, 0.5 = random, <0.5 = worse than random (invert). Threshold-independent. auROC = P(random positive ranked above random negative). On heavy imbalance, precision-recall curve often more informative.
- Single-number summaries: easy = balanced accuracy or F1; most meaningful = expected cost (assign per-error costs; fully problem-dependent).

## Lecture 04: Regression

### Big picture
- Regression = supervised learning for a quantitative (real-valued) target; predict a number, not a class. Only goal-level difference from classification.
- Four steps: simple linear regression, quality measures (RSE, R^2), multiple/OLS regression, kNN regression + parametric-vs-nonparametric.
- Symbols: x_i feature(s) of sample i, y_i true target, yhat_i prediction, w_0 intercept, w_1..w_p slopes/coefficients (ITSL uses beta), e_i = y_i - yhat_i residual, n samples, p predictors. epsilon_i = irreducible error.

### Simple (1D) linear regression
- Model: y_i = w_0 + w_1 x_i + epsilon_i; prediction yhat = w_0hat + w_1hat x.
- epsilon = catch-all for nonlinearity, omitted variables, noise.
- w_0 = intercept (yhat at x=0); w_1 = slope (change per unit x).
- PRL (population regression line, truth) vs SLR (estimate from finite sample); different samples give different SLR coefficients -> they are estimates.

### Least-squares estimation
- Residual e_i = y_i - yhat_i; RSS = sum(y_i - yhat_i)^2 = sum(y_i - w_0 - w_1 x_i)^2. Squaring: prevents cancellation, penalizes large errors (L2 -> outlier-sensitive).
- Closed form: w_1hat = sum((x_i - xbar)(y_i - ybar)) / sum((x_i - xbar)^2) = Cov(x,y)/Var(x); w_0hat = ybar - w_1hat*xbar. Line passes through centroid (xbar, ybar).
- 5 steps (memorize): (1) means xbar,ybar; (2) deviations; (3) products and squares; (4) sum and divide -> w_1; (5) back-substitute -> w_0.
- Worked (5 pts): fit yhat = 1.4 + 1.0x. Pitfall: use centered values; if sum(x_i-xbar)^2=0 slope undefined; not robust to outliers.

### Multiple / ordinary linear regression (OLS)
- Model: yhat_i = w_0 + w_1 x_i1 + ... + w_p x_ip.
- w_j = average effect of unit increase in feature j holding others fixed. Scale-dependent (normalize features); NOT a general feature-importance measure.
- Matrix: yhat = Xw; X is design matrix, one row per sample, one col per feature, leading column of ones for intercept (n x (p+1)).
- RSS = (Xw - y)^T(Xw - y); gradient = (2/n)X^T(Xw - y); normal equations X^TX w = X^T y.
- OLS closed form: w = (X^TX)^(-1) X^T y. Dim check: X^TX is (p+1)x(p+1) square, w is (p+1)x1.
- Problems/fixes: (X^TX)^(-1) may not exist (too little data, collinearity, p>n) -> Ridge (penalty on ||w||^2, also improves generalization); cost/instability -> SVD (X=U Sigma V^T; w = V Sigma^(-1) U^T y, stable, min-norm) or gradient descent (scales to big data). Multicollinearity -> near-singular, huge unstable coefficients.

### Quality measures
- RSE = sqrt(RSS/(n - p - 1)); simple regression p=1 -> denominator n-2. Typical residual size, in units of y. Divide by degrees of freedom n-p-1 (not n) because p+1 params fitted. L2 -> outlier-sensitive; complement with MAE, mean error, max residual, residual plots.
- R^2 = (TSS - RSS)/TSS = 1 - RSS/TSS, TSS = sum(y_i - ybar)^2. Proportion of variance in Y explained; relative, unitless, [0,1]. Close to 1 = good fit.
- R^2 on training never decreases when adding features -> use adjusted R^2 or test-set R^2. High R^2 does not prove correctness/causality.
- RSE vs R^2: RSE absolute/units-of-y/compare same target; R^2 relative/unitless/compare across problems. Both L2, outlier-sensitive; "good" is domain-dependent.
- Worked (5-pt): RSS=3.20, RSE~1.033, TSS=13.20, R^2~0.76.

### Parametric vs non-parametric; kNN regression
- Parametric (linear reg): fixed finite parameter set, fast, interpretable, needs less data; wrong form -> irreducible bias.
- Non-parametric (kNN): no fixed form, complexity grows with data, flexible, needs more data, harder to interpret.
- Prefer parametric when: form close to true f; few observations per predictor; interpretability. Prefer non-parametric when f unknown/nonlinear AND abundant low-dimensional data.
- kNN regression: compute distance from x_0 to all training points, take k nearest N_0, predict yhat = (1/k) sum_{i in N_0} y_i (average). Non-parametric.
- Small k = wiggly/overfit (low bias, high variance); large k = smoother; k=n predicts global mean ybar. Needs feature scaling; lazy (O(n)/query); curse of dimensionality.
- Classifier vs regressor: same algorithm, differ only in aggregation - classifier = majority vote, regressor = mean of neighbors' targets.

## Lecture 05: Logistic Regression and Bayesian Classifiers

### Big picture
- Classification = predict qualitative class label. Core idea: estimate P(Y=c | x), pick most probable class.
- Discriminative (logistic regression): model P(Y=c|x) directly. Generative (LDA/QDA/Naive Bayes): model P(x|Y=c), flip via Bayes. Both end at: predict class with highest posterior.

### Why not linear regression for classification
1. >2 classes: numeric coding (stroke=1, overdose=2, seizure=3) invents fake order and distances; different encodings give different models.
2. 2 classes: even with 0/1 coding, a line is unbounded -> predicts values outside [0,1], not valid probabilities. (Bonus: linear fits sensitive to outliers, shift boundary.)

### Logistic / sigmoid function
- p(x) = e^(w_0+w_1 x)/(1 + e^(w_0+w_1 x)) = 1/(1 + e^-(w_0+w_1 x)), always in (0,1).
- S-shaped: p->1 as z->+inf, p->0 as z->-inf; crosses 0.5 where z = w_0+w_1x = 0, i.e. x = -w_0/w_1 (decision boundary). w_1>0 rising, w_1<0 falling, larger |w_1| steeper.
- Worked (w_0=-4, w_1=0.05): x=80 -> z=0 -> p=0.5; x=100 -> z=1 -> p~0.731; x=40 -> z=-2 -> p~0.119.

### Log-odds / logit
- Odds = p/(1-p); log(p/(1-p)) = w_0 + w_1 x - logistic regression is linear in the log-odds (generalized linear model).
- Unit increase in x adds w_1 to log-odds, i.e. multiplies odds by e^(w_1).

### Fitting and decision rule
- No closed form. Maximum Likelihood Estimation: likelihood = prod_{y_i=1} p(x_i) * prod_{y_i=0}(1 - p(x_i)); take log-likelihood (= negative binary cross-entropy / log-loss); differentiate, solve numerically. Maximizing log-likelihood = minimizing binary cross-entropy.
- Decision: yhat = 1 if p(x) >= 0.5 else 0. Boundary linear (p>=0.5 iff z>=0). 0.5 is a movable default (trades FP vs FN, swept by ROC).
- Multiple features: p(x_i) = e^(w.x_i)/(1 + e^(w.x_i)), w.x_i = w_0 + sum w_j x_ij.

### K > 2 classes - TWO ways
1. One-vs-all / reference-class (multinomial): pick pivot class K; log(p(k|x)/p(K|x)) = w_k.x stays linear. p(K|x)=1/(1 + sum_{j=1..K-1} e^(w_j.x)); p(k|x)=e^(w_k.x)/(same denom). Choice of pivot doesn't change predictions.
2. Softmax coding (symmetric, no reference): p(k|x) = e^(w_k.x) / sum_{j=1..K} e^(w_j.x). softmax sigma(z)_i = e^(z_i)/sum_j e^(z_j) maps any real vector to a probability vector. Generalizes sigmoid; reappears in deep learning output layers.
- Worked softmax z=(2.0,1.0,0.1): e=7.389,2.718,1.105 sum=11.212 -> p=(0.659,0.242,0.099), predict class 1.

### Bayesian classifiers
- Bayes' theorem: P(Y=c|x) = P(x|Y=c)P(Y=c)/P(x). posterior = likelihood * prior / evidence. P(x) same for all classes -> ignore when comparing.
- Bayes classifier: yhat = argmax_c P(x|Y=c)P(Y=c). Optimal for 0-1 loss - minimizes misclassification rate; its floor is the irreducible Bayes error rate. Can't build it (needs true distributions) -> LDA/QDA/NB are approximations.
- LDA: Gaussian class-conditionals, shared covariance Sigma -> linear boundary; few params, good when data scarce.
- QDA: Gaussian, per-class covariance Sigma_c -> quadratic (curved) boundary; more flexible, needs more data, overfits more.
- Naive Bayes: features conditionally independent given class: P(Y=c|x) ~ P(Y=c) prod_j P(x_j|Y=c). Estimate p easy 1-D densities instead of one p-dim density; fast, robust in high dim (text), accurate even when independence false. Variants: Gaussian NB, Multinomial NB (counts), Bernoulli NB (binary). Zero likelihood zeros product -> Laplace/add-one smoothing.
- Worked NB (spam): prior 0.4/0.6, offer|spam=0.8, meeting|spam=0.2; email offer=1,meeting=0 -> Spam 0.256, Ham 0.018 -> P(spam|x)~0.934 -> Spam.

### ROC for probabilistic classifiers
- Output posterior P(Y=1|x); sweep threshold tau 0->1: predict 1 if posterior>=tau; compute TPR, FPR; plot (FPR,TPR). AUC summarizes ranking; diagonal AUC=0.5 random. Same recipe works for logistic regression.

## Lecture 06: Feature Engineering

### Big picture
- A model only sees features (numbers computed from data). Feature engineering turns raw data into a numeric feature matrix so a simple (usually linear) model works well.
- Default workflow: Raw data -> 1. Feature construction (numbers from everything incl. categorical) -> 2. Feature space lifting (ADD derived features: powers, products, logs) -> 3. Feature selection (REMOVE, keep the few that help).
- Lift first, then select: lifting adds expressiveness (linear-in-weights model, non-linear in original features); too many features -> curse of dimensionality / overfitting, esp. p > n. Rule of thumb: for p to grow safely, n should grow exponentially.
- Feature map: x -> phi(x). x in R^p (raw), phi(x) in R^m with m > p. Model is linear in phi: prediction = w0 + w^T phi(x).
- Correlation != causation; feature engineering cannot be fully automated ("an art"). Use residual/scatter plots, domain knowledge, experiments.

### Encoding qualitative (categorical) data
- Naively mapping labels to integers (red=0, green=1, blue=2) lies about ordering/magnitude; model exploits it.
- Boolean / 2 labels: single 0/1 (or -1/+1) feature.
- k labels, natural order: Thermometer code (cumulative ordinal) = k bits, first j entries 1.0 rest 0.0 (encodes "at least this level").
- k labels, no order/structure: One-hot code = length-k vector, all 0 except a single 1.
- Terminology: label encoding = one integer per label; ordinal encoding = label encoding respecting real order; one-hot/dummy = indicator vector.
- Dummy-variable trap: with one-hot, sum of k dummies = 1 for every row. With an intercept w0 this is perfect multicollinearity -> singular design matrix, non-unique OLS weights. Fix = dummy encoding (drop one reference category). Tree-based and Ridge/Lasso tolerate full one-hot.
- Pitfalls: high-cardinality (e.g. ZIP) blows up dimensionality (group/frequency/target encoding, embeddings); fit encoder on training data only; define fallback for unseen categories.

### Feature space lifting
- Add features computed solely from existing features; replace x with higher-dim phi(x). Keeps cheap/robust linear machinery while producing polynomial/log/curved boundaries: linear in w, non-linear in original features.
- Polynomial features: phi(x) = [1, x, x^2, x^3, ..., x^d]; d is a hyperparameter; constant 1 = intercept. yhat = w0 + w1 x + ... + wd x^d is plain linear regression on phi(x).
- General: can also add sqrt(x_j), ln(x_j), 1/x_j ("if we can add polynomials, we can add anything").
- Rules/pitfalls: ALWAYS keep the original feature when adding a modified version. Scale/standardize before lifting (x^3 for x=100 is 1e6). More features -> more overfitting -> lift then select.
- "LDA with polynomial features vs QDA": closely related, NOT identical. QDA derives quadratic boundary from per-class Gaussian covariances; LDA-on-poly fits weights on x_j^2 and cross terms directly.

### Interaction terms
- Interaction = product of two features: x_interact = x_j x_k, with own weight w_interact.
- Captures multiplicative/synergistic effects (effect of one feature depends on another). Still linear in w, no longer linear in original features. Pitfall: keep main effects x_j, x_k (hierarchy principle); pairwise terms count = C(p,2), feeds curse of dimensionality.

### Curse of dimensions
- Increasing p -> more overfitting. p > n especially dangerous (can fit training perfectly, generalize terribly). Unrelated features actively hurt test performance. Fewer features generalize better.
- Shark/correlation trap: ice-cream sales correlate with shark attacks; confounder = temperature. Selecting by marginal correlation alone is "deeply wrong"; correlation != causation.

### Feature selection (three families)
- (1) Brute force / best subset (exhaustive); (2) classifier-independent statistical filters; (3) classifier-aware wrappers/embedded.
- Best subset: try every subset (2^p - 1 non-empty), keep best. Optimal but exponential. p=5: 31 subsets. Practical only for small p.
- Filters (classifier-independent, cheap): score each feature/pair vs target.
  - Correlation (linear): rho_XY = cov(X,Y)/(sigma_X sigma_Y), in [-1,1], 0 = no linear relation.
  - Mutual information I(X;Y): general dependence, =0 iff independent.
  - Also chi-square (categorical) and ANOVA F-test (numeric vs categorical). Named methods: mRMR, Joint Mutual Information, Correlation Feature Selection.
  - mRMR = maximize relevance to target, minimize redundancy among selected. Solved via heuristics.
- Wrappers/embedded (take classifier into account): judge each feature set by actually training+testing. Quality criterion = classwise-averaged classification rate; NEVER compute on test data; best practice = separate validation set.
  - Forward selection: start empty; add the single feature that most improves the criterion; stop when no addition improves or MAXF reached. ~O(p^2); greedy.
  - Backward elimination: start with all; remove the feature whose removal hurts least; stop when any removal degrades. Better with interactions; costly for large p.
  - Stepwise (forward + backward): alternate until the selected set is unchanged; can undo greedy mistakes.
  - Embedded: RFE (rank by importance, drop weakest, retrain), Lasso L1 (penalty lambda sum|w_j| drives weights to exactly 0), tree-based importance.
- Nesting facts (ITSL 6.6 #1): forward M_k subset of M_{k+1} TRUE; backward M_k subset of M_{k+1} TRUE; cross-path containments FALSE; best-subset M_k subset of M_{k+1} FALSE. Best subset has smallest training RSS but not guaranteed best test RSS (highest overfit risk).

## Lecture 07: Outliers, Cross-Validation and Resampling

### Big picture
- Theme: how to trust a result computed from a finite, imperfect dataset.
- Golden rule (repeated): "Train and test data must have no (unnecessary) connection to each other." Source of most CV mistakes (leakage) and bootstrapping caveats.

### Outliers and high-leverage samples
- Outlier: sample whose target y_i is far from predicted yhat_i (weird in y, large residual).
- High-leverage sample: sample with very unusual feature values x_i (weird in x).
- Most dangerous = high leverage + outlier (single point can tilt the whole regression line).
- Why care: influence model parameters for the worse; dominate quality measures (RSS in test). Squaring means one residual of 10 (=100) outweighs ten residuals of 1 (=10 total).
- Detection: regression -> plot residuals r_i = y_i - yhat_i; classification -> plot distances to decision boundary; high leverage simple regression -> plot feature vs target. General advice: look at all your data, visualize before experiments.
- Z-score rule: z_i = (x_i - mu)/sigma; flag |z| > 3 (some use 2.5). Caveat: mu, sigma pulled by outliers -> masking (a big outlier hides itself; assumes near-normal).
- IQR rule (robust, distribution-free): IQR = Q3 - Q1; outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR (1.5 = Tukey; 3.0 = "far out"). Robust because quartiles barely move; better for skewed data.
- Handling: do NOT blindly remove. 1) Contact data source to check validity. 2) Remove only if "300% sure" it is incorrect AND you can reason why, then document; "results get better" is explicitly NOT sufficient. 3) Always comment on outliers even if kept. 4) ITSL: an outlier may indicate a model deficiency (missing predictor).
- Robust alternatives to deletion: MAE/L1/Huber loss, median/RANSAC/trees, winsorizing/capping to IQR fences.
- Pitfalls: removing to improve a metric = p-hacking; z-score masking on skewed data; removing legitimate rare cases hurts generalization.

### Why cross-validation (small-data problem)
- Good plain split needs: training set big enough (more data doesn't change results) AND test set big enough to span feature space. For small data these conflict.
- Terminology (slide 15): ITSL calls held-out set "validation data"; this lecture calls it "test data". Ideally want test-validation-train.
- CV solution: rotate the held-out part so every sample is used for both training and testing.

### LOOCV
- Train on all n-1 samples, test on the 1 left out, repeat n times. CV_(n) = (1/n) sum L(y_i, yhat_i^(-i)).
- Advantages: nearly unbiased (each model trains on ~all data); uses all data; DETERMINISTIC (exactly reproducible).
- Disadvantages: expensive (n fits); high variance (n training sets overlap almost entirely -> fold results highly correlated -> averaging barely reduces variance).
- OLS shortcut: CV_(n) = (1/n) sum [ (y_i - yhat_i)/(1 - h_i) ]^2 using leverage h_i.

### k-fold CV
- Split into k folds (typically k = 5 or 10); each fold is test once, others train; rotate. LOOCV = special case k = n.
- Advantages: smaller variance than LOOCV (folds overlap less), large training set, cheap (k fits). Bias-variance trade-off: small k -> more bias, less variance; large k -> low bias, high variance; 5/10 is the sweet spot.
- Dangers (misuse): everything below must happen INSIDE each fold on the training part only -- normalization/scaling, dimensionality reduction (PCA), feature selection, hyperparameter tuning. Leakage from doing these on the whole dataset inflates accuracy.
- Connected data: "left eye - right eye" problem -> use group-aware splits. Imbalance -> stratified k-fold. Time-series -> ordinary CV invalid; use forward-chaining.
- Solutions: nested CV (inner CV tunes, outer CV measures); library CV usually does NOT guard against connections -- implement yourself.

### Bootstrapping
- Motivation: one experiment = one number; want variance / confidence interval, but only one dataset.
- Assumption: dataset is a good representative of the population. Resample WITH replacement to size n; make B datasets (e.g. 1000); compute statistic on each; get variance / SE / CI.
- Bootstrap SE: SE_B(theta) = sqrt( (1/(B-1)) sum_b (theta*b - mean_theta*)^2 ). 95% percentile CI = 2.5th and 97.5th percentiles.
- ~63.2% fact: prob a specific sample never chosen = (1 - 1/n)^n -> e^-1 ~ 0.368. So ~36.8% left out (out-of-bag, OOB), ~63.2% unique appear. OOB = free test set -> Random Forest OOB error; bootstrapping = the B in Bagging (Bootstrap aggregating).
- Advantages: simple, broadly applicable, asymptotically more accurate than normality-based intervals. Disadvantages: depends on representative sample, time-consuming, problem-specific.
- Data augmentation (bonus): simulate plausible new samples (rotate/mirror/noise) mostly for images; must stay physically valid; respect train/test separation.

## Lecture 08: Optimization

### Big picture
- Training a model = an optimization problem: argmin_w E(w). E(w) = error/objective/cost/loss; w = parameter vector; argmin = the w that minimizes (vs min = the value).
- Most real problems cannot be solved analytically (can't set derivative to 0 and solve) -> iterative methods.
- Every ML tuning task is argmin_w E(w): best k for kNN, feature selection, NN topology, learning rate/batch size, SVM C and gamma, RF depth/trees.
- Two families: graph optimization (Dijkstra, BFS/DFS, A*) vs function optimization (this lecture).
- Difficulty ladder: analytical gradient -> numerical gradient -> black-box. Less info about E -> cruder, more general method.

### Gradient descent
- Iteration (memorize): w_{t+1} = w_t - lambda * grad E(w_t).
- grad E = vector of partials, points to steepest ASCENT; minus sign goes downhill. lambda = learning rate (also eta). Steeper gradient -> longer step; near a minimum gradient -> 0 so steps shrink (self-braking).
- Stop criterion: ||w_{t+1} - w_t||_2 < epsilon (also: small gradient norm, max iterations, E stops improving).
- Convergence guaranteed only under conditions: E convex (bowl, one min), grad Lipschitz, appropriate lambda. General ML E is non-convex -> finds a good, not necessarily best, solution. Random init of w0 can give different results.
- Choosing lambda: too small -> painfully slow / may stall; too large -> overshoot, oscillate, diverge. Strategies: constant; line search (Armijo-Goldstein); constant then reduce (learning-rate decay).
- Precondition: needs a computable informative gradient -> parameters w must be continuous, not discrete.
- Worked: f=theta^2, grad=2theta, theta0=4, eta=0.1 -> theta_{t+1} = 0.8 theta_t: 4 -> 3.2 -> 2.56 -> ... -> 0. With eta=1.5: diverges. Convergence needs |1 - 2eta| < 1 (0 < eta < 1); eta=0.5 reaches min in one step.
- Numeric gradient (no formula): finite difference dE/dw_i ~ [E(w + h e_i) - E(w - h e_i)] / (2h).

### Black-box optimization
- Cannot compute grad E at all; only input->output evaluation E(w) available. "Most optimization problems are of this kind."
- Categorize by three questions: 1) computational cost of one E(w) eval; 2) |w| number of parameters; 3) parameter types (discrete/continuous/mixed).

### Grid / exhaustive search
- Try every combination, always store best w. scikit-learn: GridSearchCV.
- Advantages: finds global min (within grid resolution); explainable; simple; parallelizable; reproducible/deterministic.
- Disadvantages: costly; combinatorial explosion, grows exponentially with |w| (10 values x 5 params = 10^5).
- Use: few combinations AND cheap eval; best with discrete params or |w| <= 3. "The #1 method for quick and dirty optimization."
- Multilevel (coarse-to-fine): coarse grid -> refine around best anchor -> repeat; stop when delta E < epsilon or max level.

### Monte Carlo / random search
- Randomly sample valid w, evaluate E(w), store if it beats best; repeat as long as you like, stoppable/resumable with no penalty.
- Advantages: trivially stoppable/resumable; scales to high-dim where grid explodes; great when some params matter more than others (Bergstra-Bengio argument); builds intuition when params interact.
- Disadvantages: no directed search (blind); low chance of hitting exact global min.
- Use: many interacting parameters, eval "somewhat costly." If 1 of 3 params matters, a 5x5x5 grid tests it at only 5 distinct values; 125 random draws test up to 125 distinct values.

### Advanced methods (name at least 3)
- Coordinate-wise: optimize one parameter at a time (pick coordinate randomly), any 1-D method along it. Pros: easy, can beat full grid, mix optimizers per dim. Cons: no global guarantee, bad if params interact.
- Simulated annealing (SA): temperature T; high T accepts worse moves (escape local minima), T cools toward 0 -> greedy. Metropolis: P = exp( -(E(w_new)-E(w)) / T ) for worse moves. Pros: provably reaches global optimum under conditions, no gradient. Cons: schedule/move-generation problem-dependent.
- Population/genetic (evolutionary, particle swarm, ant colony): keep a population; fitness E(w); select mating pool; crossover + mutation per generation. Cons: "a parameter mess" ("There is always a better method than genetic optimization").
- Bayesian optimization: surrogate (Gaussian process) predicts E(w) + uncertainty; acquisition function picks next point balancing exploration vs exploitation; learns from every eval. Acquisitions: Expected Improvement, UCB. Pros: reaches min in very few evals. Cons: heavy overhead. Use: eval extremely expensive, few params (<20), e.g. NN hyperparameter tuning.

### Decision summary
- Differentiable + gradient available -> gradient descent; numeric gradient -> numeric GD; few combos, cheap, discrete/|w|<=3 -> grid; many interacting params, somewhat costly -> random search; too many for grid, mixed types -> coordinate-wise; discrete signals / escape local minima -> simulated annealing; many discrete params -> genetic; very expensive eval, few params (<20) -> Bayesian optimization.
- Grid size = product of per-parameter value counts (v^k); 3 params x 5 values = 125 evals.

## Lecture 09: Advanced Classifiers (SVM, Decision Trees, Random Forests)

### Big picture
- Three off-the-shelf supervised classifiers built on the bias-variance trade-off.
- SVM = maximal margin classifier + soft margin (C) + kernels. ITSL: "one of the best out-of-the-box classifiers."
- Decision trees: interpretable, no scaling needed, but overfit and are non-robust (high variance).
- Random forests: average many decorrelated trees to crush variance; lose interpretability, recover via feature importance.

### Hyperplane and separating classifier
- Hyperplane in p dims: alpha^T x + b = 0. alpha = normal vector (orientation), b = offset.
- Classify by sign of alpha^T x + b. Labels y_i in {-1,+1}; separating means y_i(alpha^T x_i + b) > 0 for all i.
- Signed distance d = (alpha^T x + b) / ||alpha||_2. Larger |d| = more confident (heuristic, not calibrated).
- Problem: infinitely many separating hyperplanes exist -> need the margin to pick one.

### Maximum margin hyperplane (MMH) and support vectors
- Margin = minimal distance from closest points to boundary; MMH maximizes it.
- Support vectors = the few points lying exactly on the margin. MMH depends ONLY on them; delete others and boundary is unchanged.
- Formulation: max M s.t. sum alpha_j^2 = 1, y_i(alpha^T x_i + b) >= M.
- Equivalent: min (1/2)||alpha||^2 s.t. y_i(alpha^T x_i + b) >= 1. Then margin width = 2/||alpha||.
- Minimizing ||alpha|| = maximizing margin.
- Pitfall: fragility - a single new point near boundary makes MMH "jump." Real data rarely separable -> motivates soft margin.

### Soft margin and parameter C (the actual SVM)
- Slack variables xi_i let points violate the margin: min (1/2)||alpha||^2 + C*sum(xi_i) s.t. y_i(alpha^T x_i + b) >= 1 - xi_i, xi_i >= 0.
- xi_i = 0: correct outside margin; 0<xi_i<=1: inside margin but correct; xi_i>1: misclassified.
- Large C: heavy slack penalty -> near hard margin -> NARROW margin, few violations, fits hard -> OVERFIT risk, fewer SVs.
- Small C: cheap violations -> WIDE margin, more tolerance/regularization -> more robust, more SVs, may underfit.
- EXAM WARNING: lecture's (common) convention has smaller C = more regularization; ITSL defines C the opposite way. Check the convention.
- Tune C in exponential steps: {0.01, 0.1, 1, 10, 100}.
- Memory hook: Big C = Cramped margin (overfit); Small C = Spacious margin (regularize).

### Kernel trick vs feature lifting
- SVM solution: f(x) = b + sum_{i in SV} beta_i * x^T x_i. Data appears ONLY as inner products; beta_i nonzero only for SVs.
- Kernel trick: replace x^T x_i with K(x, x_i) = phi(x)^T phi(x_i), the inner product in a lifted space.
- Feature lifting (explicit): actually compute high-dim phi(x), run linear method there. Cost blows up; impossible if phi is infinite-dim.
- Kernel trick (implicit): never form phi(x); only need the inner product, which K gives directly from originals.
- Polynomial kernel degree d: K = (1 + sum_j x_j x_ij)^d. Equals explicit polynomial lifting.
- RBF/Gaussian kernel: K = exp(-gamma * ||x - x_i||^2). gamma > 0 to tune. Very LOCAL; lifted space is infinite-dim (only kernel trick works).
- gamma effect: large gamma = fast decay, wiggly boundary, OVERFIT; small gamma = smooth, may underfit. RBF-SVM has 2 knobs: C and gamma.

### Extra SVM notes
- ROC for free: threshold the distance f(x) >= t, sweep t, compute sensitivity/specificity.
- Multiclass (SVM is natively 2-class): One-vs-One (pairwise, C(n,2) classifiers, majority vote); One-vs-All (n classifiers, largest summed distance).

### Decision trees
- Segment feature space into axis-aligned rectangular regions via yes/no splits. Work for regression AND classification.
- Terminal nodes/leaves define regions. Regression prediction = mean of region; classification = most common class.
- Regression build: minimize RSS = sum_j sum_{i in R_j} (y_i - yhat_Rj)^2.
- Exhaustive partition infeasible -> recursive binary splitting: top-down + greedy (best split now, never look back).
- Each split picks feature X_j and cutpoint s: R1={X_j<s}, R2={X_j>=s} minimizing the two-region RSS.
- Stop criteria: min samples per leaf, or RSS-decrease threshold (weak - "a good split may come later").

### Classification trees: impurity measures
- Classification error rate: E = 1 - max_k(phat_mk). Not sensitive enough for growing.
- Gini index: G = sum_k phat_mk(1 - phat_mk) = 1 - sum_k phat_mk^2. Node impurity (0 = pure).
- Entropy: D = -sum_k phat_mk log2(phat_mk). Impurity in bits.
- Both small when pure, large when mixed. Choose split maximizing weighted impurity reduction (info gain for entropy).
- Example (8 A, 2 B): Gini = 1-(0.64+0.04)=0.32; entropy ~0.722 bits. 50/50 -> Gini 0.5, entropy 1 bit (max).
- Trees handle qualitative features directly (no dummies).

### Tree pruning
- Full tree overfits (leaves isolate single points). Fix: grow large T0, prune back.
- Pre-pruning (early stop): min samples/leaf, max depth, min impurity decrease. Fast, can stop too soon.
- Post-pruning (preferred): cost-complexity pruning - minimize sum_m sum_{x_i in R_m}(y_i - yhat_Rm)^2 + alpha*|T|.
- |T| = number of leaves; alpha >= 0 penalizes size. alpha=0 = full tree; larger alpha removes nodes in predictable order.
- Choose alpha via validation/cross-validation. Classification error rate usually best FOR pruning.

### Trees: pros and cons
- Pros: easy to explain/interpret (MAIN advantage), great visualization, handles qualitative predictors, no scaling needed.
- Cons: lower predictive accuracy in base form; very non-robust (small data change -> whole tree changes, high variance).

### Ensembles: bagging
- Trees have high variance. Var(mean of n iid) = sigma^2/n -> averaging reduces variance.
- Bagging: make B bootstrap datasets, train a deep unpruned tree on each, then average (regression) or majority vote (classification).
- B is not critical: past convergence more trees don't help or hurt. Result: lower variance, less overfitting.

### Random forests
- Bagged trees are highly correlated (all split on the one strong feature first) -> averaging helps little.
- Fix: at EVERY split consider only m random features out of p. Typically m ~ sqrt(p) for classification.
- Forces weak predictors in -> DECORRELATES trees -> much lower variance than bagging. Better accuracy.
- Example: p=16, m=4 -> ~75% of splits the dominant feature isn't even available.
- Key parameters (name >=4): (1) split criterion Gini/entropy/RSS; (2) stop criterion max depth / min samples per leaf / min impurity decrease; (3) number of trees B (n_estimators); (4) m = features per split (max_features, ~sqrt p). Also: bootstrap sample size, min samples to split.
- Lost interpretability -> recover via feature importance: sum the Gini/RSS reduction each feature produces across all nodes in all trees.

### Cheat sheet formulas
- Margin width = 2/||alpha||. SVM solution f(x)=b+sum beta_i K(x,x_i).
- Gini = 1 - sum phat^2. Entropy = -sum phat log2(phat). Cost-complexity = RSS + alpha|T|.
- Kernel trick = inner products only, never compute phi(x). RBF big gamma = tight/wiggly/overfit.
- Random Forest = Bagging + random features (sqrt p) to decorrelate.

## Lecture 10: Wrap-Up - An ML Project Guideline

### Big picture
- Based on Appendix B ("ML Project Checklist") of Geron, Hands-On Machine Learning. Professor adds critical "Discussion" slides.
- Meta-lesson (most repeated): you have a checklist, but ALWAYS know the REASONS WHY. It works in general, not always. Never run it blindly.
- Recurring tension: scientific objective (best achievable result, rigorous) vs business objective (make money, clear a threshold). Chasing high numbers introduces errors - classically NOT strictly separating train/validation/test.

### The 8 steps
Frame -> Get -> Explore -> Prepare -> Shortlist -> Fine-tune -> Present -> Launch.

### Step 1 - Frame the problem
- Define objective in business terms; how solution is used; current solutions; frame it (supervised/unsupervised, online/offline, classification/regression); pick performance measure; check metric aligns with business goal; comparable problems; human expertise; manual solution; list and verify assumptions.
- Trap: plain accuracy on 99%-healthy data rewards "always predict healthy" -> use precision/recall/F1/ROC-AUC for imbalance.
- Discussion: scientific = maximize metric under constraints; business = clear quality thresholds. They conflict in reality.

### Step 2 - Get the data
- List needed data/amount; document source; storage; legal + access authorization; workspace; get data; convert format WITHOUT changing data; protect/anonymize sensitive info; check size/type; SAMPLE A TEST SET, PUT IT ASIDE, NEVER LOOK (no data snooping). Automate acquisition.
- Test set locked away from the start; exploring/tuning on it = data snooping/leakage -> fantasy generalization estimate.
- Discussion: for SMALL datasets a plain split violates size rules -> use cross-validation (the typical course-project situation).

### Step 3 - Explore the data
- Work on a copy; keep a Jupyter notebook record; study each attribute (type, % missing, noise, distribution); identify target; visualize; study correlations; manual solution; promising transformations; extra data needed; document.
- Use df.describe/info, histograms, correlation heatmap, scatter plots.
- Discussion: "Know your data in- and out." Jupyter = quick and dirty sketching. As soon as you write EVALUATION code, switch to pure Python (separate computing / storing / plotting).

### Step 4 - Prepare the data
- Write a FUNCTION for every transformation - 5 reasons: (1) re-prep fresh data, (2) reuse in future projects, (3) prep test set identically, (4) prep live instances, (5) treat prep choices as hyperparameters.
- Data cleaning (fix/remove outliers - optional; fill missing or drop rows/cols); feature selection (optional); feature engineering (discretize, decompose, log/sqrt/x^2, aggregate); feature scaling (standardize/normalize).
- Scaling matters for distance/gradient methods (kNN, SVM, logistic, NN); tree-based models are scale-invariant. Fit pipeline on TRAINING data only.
- Discussion (conservative): be very cautious removing outliers ("mostly cheating"); never fill missing unless 100% sure; never drop a column unless 100% sure it carries no info.

### Step 5 - Shortlist promising models
- Train many quick-and-dirty models from different families (linear, naive Bayes, SVM, RF, NN) with standard params; compare via N-fold cross-validation reporting MEAN and STD; analyze significant variables and error types; quick feature eng; shortlist top 3-5 PREFERRING models that make DIFFERENT errors (for ensembling).
- No free lunch. Mean +/- std over folds = core CV message.
- Discussion: with enough data use a separate train/validation split (CV is the small-data remedy). Default-params testing may wrongly drop a model that shines after tuning.

### Step 6 - Fine-tune
- Tune hyperparameters with CV (transformation choices are hyperparameters too); prefer RANDOM search over grid search unless few values; Bayesian optimization for very expensive training; try ENSEMBLES; when final, measure on test set ONCE. Don't tweak after seeing test score (overfits the test set).

### Step 7 - Present
- Document; clear presentation highlighting big picture FIRST; explain why it meets business objective; present interesting points, assumptions, limitations; communicate key findings memorably.
- Discussion: "beautiful visualizations" (e.g. seaborn KDE) can violate good scientific practice - ask "does it reflect reality?"

### Step 8 - Launch, monitor, maintain
- Production-ready (unit tests); monitoring code + alerts on live performance; beware model "rot" (data drift); monitor input quality; retrain on fresh data regularly, automated.

### k-means / unsupervised (supplementary)
- No labels -> discover structure. Clustering groups similar points.
- Objective (WCSS): J = sum_i ||x_i - mu_c(i)||^2. Smaller J = tighter clusters.
- Lloyd's algorithm: choose k; init centroids (or k-means++); ASSIGN each point to nearest centroid; UPDATE each centroid to mean of assigned points; repeat until assignments stop changing. Converges to LOCAL minimum -> run multiple inits (n_init), keep lowest J.
- Choosing k - elbow method: J always decreases with k, so plot J vs k and pick the "elbow." Confirm with silhouette score.
- k-means assumes spherical, similar-size clusters, Euclidean distance -> SCALE features first; sensitive to outliers.

### Iron rules
- Split test set first; touch it ONCE at the very end; never tweak after seeing test score.
- Functionize every transformation; fit preprocessing on training data only.
- Report mean +/- std from cross-validation, not a single number.
- Small data -> cross-validation; large data -> separate validation set.
- Know WHY you do each step; never run the checklist blindly.

## Course Organization: Project and Presentation Rules

### Structure and deadlines (Prof. Markus Mayer, THD SS26)
- Project = discussions/check-ins during the semester + a final presentation. The project work IS the exam (no written exam).
- Lecturer/group registration: register in iLearn by 2026-04-26 (Sunday) 17:00 - MUST register your group to him in addition to Primus exam registration. No registration -> no dataset discussion -> no project -> failed course.
- Topic document upload: 2026-05-17 (Sunday) 17:00 via an iLearn task (PDF). Project may only proceed if the document is present before the deadline.
- Presentation slide upload: a single PDF, 2 workdays before the presentation (no online slides, no PowerPoint).
- Presentations: 8/9/10 July 2026 (Wed ITC2+ 0.27, Thu I008, Fri I006). Be present the WHOLE session (max 4 lecture hours), 15 min early.
- Repetition: project only runs in SUMMER semester, never winter. If unavailable long-term, do the ML exam in one year.

### Working groups
- Groups of 1-3 members, no more than 3.
- No mixture of students with semester > 4 and 4th semesters (visiting students form groups with AIN-B 4th semesters).
- Groups: all members must understand every part (incl. code); work shared equally; topic document has extra sentence + date + sign.

### Dataset / topic rules
- Find or create a suitable dataset (kaggle or UCI ML Repository). Must allow CLASSIFICATION or REGRESSION.
- NOT allowed: unsupervised learning (clustering), time-series data.
- License must be clear and belong to the person who posted it.
- Choosing a proper dataset is PART of the project - motivate your choice. Not too large (counterproductive), not too easy. Can create your own inspired by hobbies.
- The project is an exam; copying a Kaggle notebook without original work / full understanding cannot pass; a duplicate = automatic "not passed."

### Goal / what to deliver
- Perform a proper, scientifically grounded ML EVALUATION. Not a recommendation system, chatbot, or assistant. No GUI/user interaction needed.
- Document evaluation steps and reasoning in slides + presentation - these form the main base of the grade.
- Code is NOT handed in or graded, but every part must be understood by all members. Generating whole code with an LLM without understanding -> noticed -> "not passed."
- No mandatory check-in, but strongly recommended to discuss with him at least once (best in the middle of the work).

### Dataset discussion (mandatory meeting)
- Must be lecturer-registered to get a slot; schedule announced on iLearn Mon 2026-04-27, meetings same week.
- Bring up to 3 possible datasets (prioritized) + project topics. Mindset: "Convince me this is a suitable project."
- For groups: all members must be present. No grading; if agreed, proceed; if not, repeat 1 week later. Discussion is public.

### Method requirements (minimum)
- 1 person: base classifier/regressor + 1 advanced classifier/regressor with hyperparameter optimization.
- Group of 2: base + 2 advanced with hyperparameter optimization.
- Group of 3: base + 2 advanced with THOROUGH hyperparameter optimization.

### Presentation
- Times (without questions): 1 member = 10-15 min; 2 = 15 min; 3 = 20 min. Groups: roughly equal speaking time each.
- Questions afterward (not time-limited), by two examiners; may cover general ML knowledge (mainly from the "Repetition questions" at end of slide sets), especially if you never discussed your project during the semester.
- Scientific presentation outline (like a paper/BA): Motivation, State-of-the-Art (optional), Data, Method, Results and Discussion, Outlook (optional), Summary. Keep a guiding "red line."
- Data slide: max a handful of plots; always state number of samples, features, classes, class distribution.
- Method vs Results must be SEPARATE (explain what you want to do, then show/discuss outcome - no mixture).

### Grading and common pitfalls
- No fixed grading scheme; grade mainly driven by your REASONING ("Why?") for each step. Must have a slide on data environment/motivation and how the dataset was generated.
- Breaking deadline/time rules REDUCES grades. Going overtime OR under (>1 min short) hurts. Practice with a stopwatch.
- Form pitfalls (from anonymized examples): avoid screenshots (recreate tables); avoid mass-accumulations of numbers; max 3 digits after the decimal; mark the most important number/result in tables; have slide HEADERS and slide NUMBERS on every slide; adjust font sizes so even the last table row is readable; avoid crowded slides (~5 items or a plot with 2 items); no more than 2 plots per page; don't re-explain lecture basics; avoid showing code (explain construction instead).
- Good examples: perfectly visualized diverse dataset; show image/signal samples; clean layout with header, number, one plot and few text items understandable from afar.
- Put name(s) on header slide and introduce yourself. Start early. Only use tools/methods discussed in the lecture (else you must explain them in detail).
