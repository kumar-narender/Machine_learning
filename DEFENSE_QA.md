# Presentation Defense: Consistency Check and Question Bank

Group 31 (Parth Navadiya, Narender Kumar) - Heart Disease Prediction, BRFSS 2015.
Cross-checked against Prof. Mayer's lectures (see COURSE_SUMMARY.md) and the project
notebook + draft.pptx. Prof. Mayer grades mainly on your REASONING ("Why?") and the
examiners can ask general ML questions from the repetition sets, so this file covers
both project-specific and general-theory questions.

--------------------------------------------------------------------------------
## PART A. Consistency verdict (fix these BEFORE the presentation)

### What is consistent with the course (defend confidently)
- Split BEFORE scaling and feature selection so train and test have no unnecessary
  connection (Lecture 03b golden rule). Done correctly.
- StandardScaler fitted on TRAINING data only, then applied to test (the exact "TRAP"
  from Lecture 03b). Done correctly.
- Stratified split keeps the ~10:1 class ratio in both sets (imbalance handling).
- Imbalance-aware evaluation: accuracy plus balanced accuracy, sensitivity, specificity,
  ROC-AUC, and absolute + normalized confusion matrices (Lecture 03b core).
- Feature selection uses BOTH families from Lecture 06: a wrapper (forward selection with
  the class-wise averaged rate as the criterion) and a filter (mutual information).
  Reasoned decision to keep all 21 features because p << n (no curse of dimensionality).
- Base + 2 advanced models with hyperparameter optimization via GridSearchCV = exactly the
  method requirement for a group of 2.
- Resampling (undersampling) done on TRAINING data only; test keeps the real 10:1 ratio.
- 5-fold CV reported as mean +/- std (Lecture 07 / Lecture 10 iron rule).

### BUG to fix (a real inconsistency in the code)
- In the base kNN test evaluation, the model is FIT on `X_train_s` (scaled) but PREDICTS on
  `X_test` (raw, unscaled): `knn.fit(X_train_s, y_train)` then `knn.predict(X_test)`.
  The sklearn warning "X has feature names, but KNeighborsClassifier was fitted without
  feature names" is the symptom. Because kNN is distance-based, querying scaled-space
  neighbours with raw values produces invalid predictions - the reported base kNN numbers
  (Acc 0.886, Sens 0.05) are unreliable. FIX: predict on `X_test_s`. The undersampling
  demo cell already does this correctly, so the two cells contradict each other.

### Slide-vs-code gaps (a question here has no code to point to)
1. The whole "class_weight = balanced" story (slides 18-21, 26, 27, 30, 31; Sens jumps
   from 5% to 79%; best model = balanced Linear SVM) is NOT in the notebook. There is no
   class-weighted RF or SVM cell. This is the biggest risk: if an examiner says "show me
   the code," you cannot. FIX: add `class_weight="balanced"` RandomForest and LinearSVC
   experiments to the notebook so the slides are backed by code, OR remove those slides.
2. Best-model contradiction: notebook concludes Random Forest (all features); slides
   conclude balanced Linear SVM. Pick one story and make both agree.
3. Split ratio: slide 12 says 80/20; the notebook uses test_size=0.30 (70/30). Fix the slide.
4. Pipeline slide 12 says "Feature Selection: selected informative features," but the
   notebook KEEPS all 21 features (selection made results worse). Reword to "evaluated
   feature selection; kept all features because it did not help."
5. Threshold tuning (slide 22) and Precision-Recall curve (slide 24) are described but not
   computed in the notebook. Either add the code or soften the slides to "conceptual."
6. "Feature Importance" (slide 25) lists the mutual-information / correlation ranking, not a
   model's importances. If you call it "feature importance," be ready to say it comes from
   mutual information, or add RandomForest `feature_importances_`.

--------------------------------------------------------------------------------
## PART B. Project-specific questions (grouped)

For each: the question, and R = your readiness (OK = code supports it; RISK = gap/fix needed).

### B1. Problem framing and dataset
- What is the prediction task and why is it classification, not regression? (R: OK - binary
  target HeartDiseaseorAttack, cannot meaningfully average two labels.)
- Why this dataset; is the license clear? (R: OK - Kaggle BRFSS 2015, CC0 public domain.)
- What does one row represent, how many samples (n) and features (p)? (R: OK - 253,680 rows,
  22 cols; after dedup 229,781; 21 features + target.)
- Is the data survey/self-reported, and what does that imply for validity? (R: OK - name it
  as a limitation: no clinical/lab measurements.)
- Why is choosing the dataset "part of the project"? What did you have to justify? (R: OK.)

### B2. Data cleaning and duplicates
- You removed 23,899 duplicate rows. How do you KNOW they are true duplicates and not
  different people who gave identical answers? (R: RISK - with 21 mostly-binary features,
  identical rows can be distinct people. Lecture 07 says do not blindly remove, be "300%
  sure." Prepared answer: you removed them to prevent the SAME row landing in both train and
  test (a train/test connection); acknowledge the trade-off and that you checked it did not
  distort the class ratio.)
- Any missing values? How handled? (R: OK - none missing.)
- Did removing duplicates change the class balance? (R: OK if you can state it stayed ~10%.)

### B3. Exploratory data analysis
- Why histograms and one scatter plot, and why only BMI vs Age as a scatter? (R: OK - a
  scatter needs both axes to spread; binary flags collapse to stripes. You demonstrated this.)
- What did the correlation-with-target chart show? (R: OK - GenHlth, Age, DiffWalk, HighBP,
  Stroke, HighChol, Diabetes.)
- Correlation measures only linear relationships - how did you address that? (R: OK - you
  added mutual information, which captures non-linear dependence.)
- Why look at class-conditional histograms with density=True? (R: OK - shapes comparable
  despite 10:1 imbalance.)

### B4. Train/test split and leakage
- What split ratio and why stratified? (R: RISK - code is 70/30, slide says 80/20; fix.)
- Why split before scaling and selection? (R: OK - avoid train/test connection / leakage.)
- Where exactly could leakage have crept in and how did you prevent it? (R: OK - scaler and
  selection fit on training only; selection ran on a training subsample.)
- Is a single split enough? (R: OK - you also did 5-fold CV; mention mean/std.)

### B5. Feature scaling
- Why scale for kNN but not strictly for Random Forest? (R: OK - kNN/SVM are distance/
  margin based; trees are scale-invariant.)
- Which features forced scaling? (R: OK - BMI range 86, PhysHlth/MentHlth 30, Age 12; the
  rest are binary.)
- Does StandardScaler change the distribution shape? (R: OK - no, only recenters to mean 0
  std 1; a skewed feature stays skewed. You showed the before/after histograms.)
- Did you fit the scaler on train or all data, and why does it matter? (R: OK - train only,
  else test distribution leaks.)

### B6. Feature selection
- Name the two families you used and how they differ. (R: OK - wrapper forward selection
  (uses the classifier, greedy) vs filter mutual information (classifier-independent).)
- What criterion did forward selection optimize, and on which data? (R: OK - class-wise
  averaged rate, via CV on a training subsample; never on test.)
- Why did you keep all 21 features if selection is supposed to help? (R: OK - forward
  selection plateaued and p << n, so no curse of dimensionality; selection actively hurt
  RF and SVM (lower balanced accuracy, sensitivity, AUC). Defensible.)
- Why use LDA as the inner estimator for selection, not kNN? (R: OK - speed; LDA is a fast
  linear proxy for the search.)
- Forward vs backward vs best-subset selection - trade-offs? (R: OK - see COURSE_SUMMARY
  Lecture 06: greedy, may miss features useful only in combination; best subset is 2^p.)

### B7. Class imbalance and error measures
- Why is accuracy misleading here? (R: OK - always-predict-No gives ~90% accuracy, catches
  zero patients.)
- Define sensitivity, specificity, balanced accuracy, precision, F1. (R: OK - see
  COURSE_SUMMARY Lecture 03b. Balanced accuracy = (Sens+Spec)/2 for 2 classes.)
- What is balanced accuracy and why is it your main metric? (R: OK.)
- Give two criticisms of F1. (R: OK - ignores TN; asymmetric / depends on which class is
  positive; depends on class ratio.)
- What is the confusion matrix telling you for the base models? (R: OK - almost everything
  predicted No; high specificity, near-zero sensitivity.)
- Three standard ways to handle imbalance? (R: OK - right metric; resampling
  (under/oversampling); class weighting. NOTE: class weighting is only on slides, not code.)
- Why is undersampling done on training only? (R: OK - test must reflect the real 10:1 rate.)
- Under- vs over-sampling (e.g. SMOTE) trade-offs? (R: OK - undersampling discards majority
  data; oversampling/SMOTE synthesizes minority; you listed SMOTE as future work.)

### B8. kNN (base classifier)
- How does kNN classify a point; is it parametric? (R: OK - majority vote of k nearest by
  Euclidean distance; non-parametric, lazy, stores training data.)
- How did you choose k and what value? (R: OK - CV class-wise rate over k in {3,5,9,...,45};
  best k=3.)
- k=3 is very flexible - is that overfitting? Did you try k=1? (R: RISK - be ready: small k
  = low bias/high variance; the CV rate was highest at the smallest k tried, so the class
  signal is weak; k=1 would be even more overfit. Honest answer: the model is limited here.)
- What are kNN's weaknesses on this data? (R: OK - needs scaling, slow prediction on 160k
  rows, curse of dimensionality, poor on imbalance.)
- Does kNN output probabilities for ROC? (R: OK - predict_proba = fraction of neighbours in
  the positive class.)

### B9. Random Forest (advanced 1)
- What is a Random Forest and how does it differ from bagging? (R: OK - bagging + a random
  subset of m features per split to decorrelate trees; m ~ sqrt(p).)
- Why does averaging many trees reduce error? (R: OK - variance reduction; Var of a mean of
  n iid = sigma^2/n; decorrelation makes it work.)
- Which hyperparameters did you tune and what were the best values? (R: OK - n_estimators,
  max_depth, max_features; best = 100 trees, depth None, sqrt. NOTE: your markdown text also
  mentions min_samples_split/leaf but the actual grid did not include them - fix the text.)
- Why did optimization not beat the default? (R: OK - GridSearch picked near-default values;
  optimization is not guaranteed to improve, it confirms the default is good.)
- How do you read feature importance and what were the top features? (R: OK-ish - if the
  slide's importances are from mutual information, say so; or add RF feature_importances_.)
- Do trees need feature scaling? (R: OK - no, axis-aligned splits are scale-invariant.)

### B10. Linear SVM (advanced 2)
- What does an SVM optimize; what is the margin and a support vector? (R: OK - the maximal-
  margin separating hyperplane; support vectors are the points on the margin.)
- What does C do; which convention are you using? (R: RISK - be explicit: in sklearn LARGE C
  = less regularization = narrower margin (fits harder); small C = wider margin. The lecture
  sometimes uses the opposite convention - state which you mean. See the exam warning in
  COURSE_SUMMARY Lecture 09.)
- Why LinearSVC and not an RBF kernel SVM? (R: OK - efficiency on 160k x 21; same maximal-
  margin principle from the lecture; RBF is future work.)
- Best C found? (R: OK - C=1, equal to the default; no improvement, and you explain why.)
- decision_function vs predict_proba for ROC - is using distances valid for AUC? (R: OK -
  AUC only needs a ranking; the signed distance to the hyperplane is a valid score.)
- Kernel trick vs feature lifting - explain. (R: OK - see COURSE_SUMMARY Lecture 09: kernel
  computes inner products in a lifted space without ever forming phi(x).)

### B11. Hyperparameter optimization
- What is GridSearchCV doing internally, and on what data? (R: OK - exhaustive grid with
  inner CV on training data; test untouched.)
- Grid vs random vs Bayesian search - when each? (R: OK - Lecture 08: grid for few/cheap/
  discrete, random for many interacting params, Bayesian for very expensive evals.)
- Why balanced_accuracy as the scoring metric in the grid? (R: OK - imbalance; accuracy
  would favour the majority class.)
- Is a hyperparameter the same as a parameter? (R: OK - hyperparameter set before training
  (k, C, n_estimators); parameter learned during training (SVM weights, tree splits).)

### B12. Cross-validation
- Why 5-fold and not LOOCV? (R: OK - Lecture 07: LOOCV is expensive and high variance;
  5/10-fold is the sweet spot.)
- What do the mean and std of the CV scores tell you? (R: OK - mean = generalization
  estimate, low std = stable across folds.)
- What must happen INSIDE each fold to avoid leakage? (R: OK - scaling, selection, tuning
  must be refit per fold on the training part only.)
- Is your data "connected" in a way that needs group-aware CV? (R: OK - survey rows are
  independent people, so plain stratified CV is fine; contrast with the left-eye/right-eye
  example from the lecture.)

### B13. ROC, thresholds, PR curve
- How is a ROC curve built and what is AUC? (R: OK - sweep the threshold, plot TPR vs FPR;
  AUC = probability a random positive outranks a random negative; 0.5 = random.)
- Why might you tune the decision threshold in a medical setting? (R: OK conceptually -
  lower threshold raises sensitivity at the cost of specificity; but no threshold-sweep code
  exists, so keep the answer conceptual or add the code.)
- When is a Precision-Recall curve preferred over ROC? (R: OK - heavy imbalance; focuses on
  the positive class. NOTE: PR curve is on the slide but not in code.)

### B14. Results and final model choice
- Your best models have low sensitivity - is this project a failure? (R: OK - frame it: the
  scientific finding is that cheap survey features have limited signal; you correctly used
  imbalance-aware metrics to EXPOSE that rather than hide it behind 90% accuracy.)
- Which model do you recommend and why? (R: RISK - decide: notebook says RF (all features),
  slides say balanced SVM. Make them agree, then justify by balanced accuracy / sensitivity.)
- Why is high accuracy misleading here (tie back to the confusion matrix)? (R: OK.)

--------------------------------------------------------------------------------
## PART C. General ML theory questions (from the repetition sets - likely if you
did not discuss your project during the semester)

- What are the three types of machine learning? Which is yours? (Supervised classification.)
- Supervised vs unsupervised vs reinforcement - one line each.
- Classification vs regression - how do you tell them apart? (Can you average two outputs?)
- What is overfitting vs underfitting; how does flexibility relate to bias and variance?
- Define the bias-variance trade-off; where does kNN's k sit on it?
- What is the curse of dimensionality; why did it justify keeping all features here?
- Parametric vs non-parametric models - give one example of each from your project.
  (kNN non-parametric; LDA/SVM/logistic parametric.)
- What is the Bayes classifier and the Bayes error rate; why is it "optimal"?
- Bayes theorem: name posterior, likelihood, prior, evidence; why does the evidence drop out?
- LDA vs QDA vs Naive Bayes - assumptions and boundary shape (linear vs quadratic).
- What is a decision boundary; what shape does kNN produce?
- Why can't you use plain linear regression for classification?
- Logistic regression: what is the sigmoid, the log-odds, and how is it fitted (MLE)?
- What is a statistic; what are the two foundations of ML (statistics + optimization)?
- Gradient descent: write the update rule; what does the learning rate do if too large/small?
- What is a loss/error function; give the 0-1 error and MSE.
- Feature engineering: encoding categoricals (one-hot, the dummy-variable trap), feature
  lifting (polynomial features), interaction terms.
- Outliers vs high-leverage points; the z-score rule and the IQR rule; when do you remove one?
- Bootstrapping: what is it, the ~63.2% fact, and its link to bagging / out-of-bag error.
- What is data leakage; give three concrete ways it happens.
- Why report mean +/- std rather than a single number?
- No-free-lunch: why try several model families?

--------------------------------------------------------------------------------
## PART D. Presentation-form questions and pitfalls (Prof. Mayer's rules)

- Keep a guiding "red line" through Motivation, Data, Method, Results+Discussion, Summary.
- Data slide must state n samples, p features, number of classes, class distribution. (You do.)
- Keep Method and Results SEPARATE (explain what you will do, then show outcomes).
- No screenshots of tables (recreate them); mark the most important number; max 3 decimals;
  slide headers and slide numbers on every slide; do not re-explain lecture basics; avoid
  showing raw code (explain the construction instead).
- Timing: group of 2 = about 15 minutes; going over OR more than 1 minute under hurts the
  grade. Practise with a stopwatch. Equal speaking time each.
- Every member must be able to answer questions about ANY part, including the code.

--------------------------------------------------------------------------------
## PART E. Priority fix list before the presentation

1. Fix the kNN scaling bug: predict on X_test_s (scaled), not X_test.
2. Add the class_weight="balanced" RandomForest and LinearSVC experiments to the notebook so
   the slides 18-31 are backed by code (or cut those slides).
3. Reconcile the final model: pick RF or balanced SVM and make notebook + slides agree.
4. Fix slide 12: split is 70/30 (not 80/20); reword the "feature selection" line to say you
   evaluated it and kept all features.
5. Add threshold-tuning and a Precision-Recall curve to the notebook, or mark slides 22 and
   24 as conceptual.
6. Clarify slide 25 "feature importance" source (mutual information, or add RF importances).
7. Fix the RF markdown text to list only the hyperparameters actually in the grid.

--------------------------------------------------------------------------------
## PART F. Model answers to the hardest questions (say it in your own words)

These are the questions most likely to expose a group that does not understand its own
project. Each answer is a defensible script; the exact numbers are in the notebook run.

- "Your accuracy dropped from 90% to 74% after balancing. Did you make the model worse?"
  No. On a 10:1 imbalanced problem, 90% accuracy is what you get by always predicting
  "healthy" - it finds zero patients. The right metric is balanced accuracy (the average of
  sensitivity and specificity). Balancing raised balanced accuracy from about 0.52 to about
  0.76 and sensitivity from about 0.05 to about 0.79. The model got much better at the job
  that matters (finding sick people); accuracy only fell because we stopped rewarding the
  majority-class shortcut.

- "Why class weighting instead of undersampling or SMOTE?"
  Class weighting keeps all the training data and simply makes each class contribute equally
  to the loss, so we do not throw away majority rows (as undersampling does) or synthesize
  fake minority rows (as SMOTE does). We show an undersampling demo for kNN too, because kNN
  has no class_weight option. SMOTE is listed as future work.

- "Why did hyperparameter optimization not help?"
  GridSearchCV searched the grid and returned values essentially equal to the defaults
  (RF: 100 trees, depth None, sqrt features; SVM: C = 1). Optimization is not guaranteed to
  improve a model - it confirms which configuration in the search space is best. Ours was
  already near-optimal, which is itself a finding.

- "Why did feature selection make things worse?"
  We have about 230k samples and only 21 features (p much smaller than n), so there is no
  curse of dimensionality to fight - every feature carries a little signal. Dropping features
  removed usable information, so balanced accuracy, sensitivity and ROC-AUC all fell. That is
  why we kept all 21 features. We still ran two selection methods to justify the choice.

- "Why did you remove 24k duplicate rows - could they be different people?"
  With 21 mostly-binary survey answers, identical rows are plausible, but our main reason is
  the train/test separation rule from the lecture: if the same row appears in both train and
  test it secretly connects the two sets and inflates the score. We checked that removing them
  left the class ratio unchanged (still about 10% positive). We can show the model both ways
  if asked.

- "Is k=3 for kNN not overfitting?"
  Small k means low bias, high variance, so yes, k=3 is a flexible model. We did not pick it
  by hand - cross-validation on the training data gave the highest class-wise rate at the
  smallest k, and the score fell monotonically as k grew. On this data the neighbourhood
  signal is weak, so no k made kNN a strong classifier; that is why we moved to the advanced,
  class-weighted models.

- "Which C convention are you using?"
  scikit-learn's: large C means less regularization and a narrower margin (fits the training
  data harder); small C means more regularization and a wider margin. The lecture sometimes
  states it the other way, so we are careful to say "sklearn C." GridSearch chose C = 1.

- "You used decision_function, not probabilities, for the SVM ROC. Is that valid?"
  Yes. ROC-AUC only needs a ranking of samples, and the signed distance to the hyperplane is a
  valid ranking score. LinearSVC does not output calibrated probabilities, so we use the
  distance directly.

- "How do you know there is no data leakage?"
  The test set is split off first and never touched until the final evaluation. The scaler is
  fit on training data only; feature selection runs on a training subsample only; GridSearchCV
  and cross-validation refit everything inside each fold. Duplicates were removed before the
  split so no row can appear in both sets.

- "Why is ROC-AUC almost unchanged by balancing while accuracy changes a lot?"
  ROC-AUC measures ranking quality (the probability a random positive outranks a random
  negative) and is threshold-independent, so re-weighting classes barely moves it. Accuracy is
  measured at a single threshold and is dominated by the majority class, so it moves a lot.
  This is a good illustration of why we report several metrics.

--------------------------------------------------------------------------------
## PART G. One-line facts to memorize (fill the final column from the notebook run)

- Dataset: BRFSS 2015 (CDC survey), via Kaggle, CC0. 253,680 rows x 22 cols, all numeric,
  no missing values. After removing 23,899 duplicates: 229,781 rows, 21 features + 1 target.
- Target: HeartDiseaseorAttack (0 = no, 1 = yes). Positive share about 10% (roughly 10:1).
- Split: 70% train / 30% test, stratified, random_state = 42.
- Scaling: StandardScaler, fit on training data only. Needed for kNN and SVM, not for trees.
- Base: kNN, k chosen by 3-fold CV on balanced accuracy (best k = 3).
- Advanced 1: Random Forest (n_estimators, max_depth, max_features tuned by GridSearchCV).
- Advanced 2: Linear SVM (C tuned by GridSearchCV over {0.01, 0.1, 1, 10, 100}; best C = 1).
- Imbalance handling: right metric (balanced accuracy) + class_weight="balanced" + a kNN
  undersampling demo. Undersampling/weights on training only; test kept at the real ratio.
- 5-fold CV of kNN accuracy: mean 0.8775, std 0.0019 (stable across folds).
- Final model: Linear SVM with class_weight="balanced" (best balanced accuracy + sensitivity
  + ROC-AUC); Random Forest (Balanced) a close second with feature importances.
- Top features (correlation + mutual information + RF importance agree): GenHlth, Age, HighBP,
  HighChol, DiffWalk, Stroke, Diabetes.

--------------------------------------------------------------------------------
## PART H. If you get a question you cannot answer

- Do not bluff. Say what you do know, state your assumption, and offer to show the relevant
  cell. Examiners reward honest reasoning over confident guessing.
- Tie any answer back to the course principle behind it (train/test separation, the right
  metric for imbalance, bias-variance, the reason you scaled or did not).
- If asked "what would you do differently," use the Future Work list: SMOTE/oversampling,
  XGBoost or other ensembles, an RBF-kernel SVM, external-dataset validation, and adding real
  clinical measurements (ECG, blood tests) rather than survey answers only.

--------------------------------------------------------------------------------
## PART I. Finalized results from the fresh run (memorize these) + two subtle points

Numbers below are the actual output of the executed notebook (70/30 split, seed 42).

| Model                     | Acc   | Bal-Acc | Sens  | Spec  | ROC-AUC |
|---------------------------|-------|---------|-------|-------|---------|
| kNN (base, k=3)           | 0.877 | 0.571   | 0.185 | 0.956 | 0.666   |
| Random Forest (Default)   | 0.892 | 0.545   | 0.108 | 0.982 | 0.804   |
| Random Forest (Balanced)  | 0.868 | 0.634   | 0.340 | 0.928 | 0.808   |
| Linear SVM (Default)      | 0.899 | 0.523   | 0.050 | 0.996 | 0.836   |
| Linear SVM (Balanced)     | 0.736 | 0.761   | 0.792 | 0.730 | 0.837   |

Final model = Linear SVM (Balanced): best balanced accuracy (0.761) and sensitivity (0.792).
5-fold CV of base kNN: mean accuracy 0.8775, std 0.0018.

### Subtle point 1: why balancing helped the SVM far more than the Random Forest
- Class weighting lifted the Linear SVM from 0.05 to 0.79 sensitivity, but the Random Forest
  only from 0.11 to 0.34. Reason: class_weight reweights the training loss; a linear SVM
  moves its single hyperplane a lot in response, whereas each RF tree still votes by majority
  in its leaves, so the ensemble shifts less. If asked "why not the balanced RF," this is the
  answer, and it is exactly why you chose the balanced SVM. (You could push RF sensitivity up
  with undersampling or threshold tuning - list as future work.)

### Subtle point 2: RF feature importance disagrees with the correlation/MI ranking
- Correlation and mutual information rank GenHlth, Age, HighBP, HighChol, DiffWalk highest.
- RF impurity importance ranks BMI, Age, Income, GenHlth, PhysHlth highest.
- They differ because impurity-based importance is biased toward high-cardinality / continuous
  features (BMI has 84 values, Income 8), which offer more split points, while correlation/MI
  measure association with the target directly. Both are valid; mention the bias if asked why
  BMI and Income jump up in the RF chart. (The slide "Feature Importance" now shows the RF
  ranking; the earlier EDA slide shows the correlation ranking - be ready to explain both.)

### Status of the Part E fixes (all applied in this push)
1. kNN scaling bug fixed (predicts on X_test_s); base kNN is now 0.877/0.571/0.185/0.666.
2. class_weight="balanced" RF and SVM added and executed.
3. Final model reconciled: notebook and slides both pick balanced Linear SVM.
4. Slide split corrected to 70/30; feature-selection line reworded; sample/feature counts and
   data-collection source added to the data slide (all in presentation_final.pptx).
5. Threshold-tuning and precision-recall curve added to the notebook.
6. RF feature_importances_ added; slide 25 updated to the real RF ranking.
7. Comparison table is now computed from real predictions, not hardcoded.
