## Setup
1. Download `data.csv` and `compiled.csv` from https://osf.io/pyb8s into this folder (only needed to rerun 01_cleaning.ipynb; both are too large for GitHub)
2. pip install -r requirements.txt
3. Run the notebooks in order: 01_cleaning.ipynb → task1.ipynb → task2.ipynb
   (task1 and task2 load the cleaned .parquet files already in the repo, so they run without step 1)


Our two questions
- Task 1 (Person B): can we predict whether an English learner reaches near-native grammar proficiency using only their language background?
- Task 2 (Person C): can we predict a learner's native language from which grammar questions they get right or wrong? This shows whether different language backgrounds make different kinds of mistakes.


Shared settings (everyone uses these)
- RANDOM_STATE = 42
- Train/test split: 80/20, stratified on the target
- Scaling and one-hot encoding go inside a scikit-learn Pipeline, fit on training data only
- Tune on a stratified sample (for example 50,000 rows) if a model is slow, then retrain the best model on the full training set

---------------------------------------------------
1. **Checked the data:** confirmed each person's score matches their 95 right/wrong answers.
2. **Removed junk:** 5 duplicate people, plus columns that were blank, identical for everyone, or just bookkeeping (dates, times).
3. **Defined "proficient":** a learner is proficient if they score at least 0.958. That's the score 75% of native English speakers beat, so a proficient learner does as well as at least a quarter of native speakers. About 37% of learners qualify.
4. **Kept only English learners** (non-native speakers) for Task 1.
5. **Cleaned education** into a 1–6 scale (1 = didn't finish high school, 6 = graduate degree).
6. **Filled in blanks:** about 3% of learners skipped the questions about time abroad; we filled in the typical answer (0) and flagged who skipped.
7. **Fixed a broken column:** "English is a primary language" was wrongly 0 for everyone, so I rebuilt it.
8. **Created new features:** years of English exposure, starting-age group, share of time spent abroad, number of languages spoken, whether they live in an English-speaking country, and language family.
9. **Explored the data with charts:** starting English younger and living abroad longer both go with higher proficiency, and native language matters too.
10. **Saved two files:** `clean_learners.parquet` (Task 1) and `clean_native_lang.parquet` (Task 2).

---------------------------------------------------
Task 1: proficiency prediction 

Question: is this learner near-native in English grammar?

- File: clean_learners.parquet
- Target: proficient (1 = yes, 0 = no). About 37% are 1, so the classes are imbalanced.
- Inputs: background columns only
- Numeric: age, Eng_start, years_exposure, Eng_country_yrs, lived_eng_share, house_Eng, background_missing, education_level, psychiatric, prime_Eng, lives_eng_country, n_native_langs, n_prime_langs
- Categories (one-hot encode): gender, start_age_group, native_lang_grp, lang_family

Don't use as inputs: score (the label is calculated from it, so using it is leakage) and id
Models: logistic regression, kNN, decision tree, random forest, gradient boosting
Tuning: cross-validated grid or random search
Imbalance: class_weight="balanced" or resampling
Metrics: F1 and ROC-AUC (main), plus precision, recall, confusion matrix. Accuracy alone is misleading here.
Analysis: model comparison table, feature importance (which background factors matter most), who gets misclassified

---------------------------------------------------
Task 2: native-language prediction

Question: what's this learner's native language, based on their grammar mistakes?

- Notebook: task2.ipynb
- File: clean_native_lang_132.parquet
- Target: native_lang, one of 10 languages: Finnish, Turkish, German, Russian, Hungarian, Polish, Dutch, Swedish, Spanish, Portuguese
- Inputs: 132 grammar questions (0/1)
  - 95 scored `q` columns (the ones native speakers agree on)
  - 37 extra `d_q` columns (the ones native speakers disagree on, which the original authors left out of the score)
- Don't use as inputs: id

Class sizes are uneven: Finnish ~40K down to Portuguese ~10K, so use class weights
Models: logistic regression, Naive Bayes, random forest, gradient boosting, MLP, and a soft-voting ensemble
Tuning: 3-fold GridSearchCV on a 50,000-row stratified sample, scored by macro F1
Metrics: macro F1 (main), accuracy, macro precision and recall, confusion matrix
Experiments: 95 vs. 132 questions; adding skill information (+ score, skill-adjusted answers)
Analysis: which languages get confused, accuracy by learner skill level, which questions matter most, language "signatures", and language-family prediction

Results: the voting ensemble was best (55.8% accuracy, macro F1 0.531, vs. a 17.8% baseline). Adding the 37 extra questions raised macro F1 by about 0.13. Most mix-ups are between related languages (Portuguese/Spanish, Dutch/German, Polish/Russian).

## Setup
1. Download data.csv from https://osf.io/pyb8s into this folder (only needed to rerun 01_cleaning.ipynb)
2. pip install -r requirements.txt
3. Run the notebooks in order: 01_cleaning.ipynb → task1.ipynb → task2.ipynb