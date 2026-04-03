import json

with open('COMP4040_Lab6_Classification_Regression.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# TODO: Train a DecisionTreeClassifier with max_depth=1 and call results(clf)" in source:
            cell['source'] = [
                "# TODO: Train a DecisionTreeClassifier with max_depth=1 and call results(clf)\n",
                "clf_depth1 = DecisionTreeClassifier(max_depth=1, random_state=42)\n",
                "results(clf_depth1)"
            ]
        elif "# TODO: Try at least two more depth values and observe how the boundary changes" in source:
            cell['source'] = [
                "# TODO: Try at least two more depth values and observe how the boundary changes\n",
                "for depth in [3, 10]:\n",
                "    print(f'\\nMax depth: {depth}')\n",
                "    clf_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)\n",
                "    results(clf_depth)"
            ]
        elif "# TODO: BaggingClassifier with n_estimators=1" in source:
            cell['source'] = [
                "# TODO: BaggingClassifier with n_estimators=1\n",
                "bagging_1 = BaggingClassifier(DecisionTreeClassifier(max_depth=30, random_state=42), n_estimators=1, random_state=42)\n",
                "results(bagging_1)"
            ]
        elif "# TODO: BaggingClassifier with more estimators (try at least 2 values)" in source:
            cell['source'] = [
                "# TODO: BaggingClassifier with more estimators (try at least 2 values)\n",
                "for n_est in [10, 50, 100]:\n",
                "    print(f'\\nBaggingClassifier with n_estimators: {n_est}')\n",
                "    bagging_n = BaggingClassifier(DecisionTreeClassifier(max_depth=30, random_state=42), n_estimators=n_est, n_jobs=-1, random_state=42)\n",
                "    results(bagging_n)"
            ]
        elif "# TODO: RandomForestClassifier with max_depth=30" in source:
            cell['source'] = [
                "# TODO: RandomForestClassifier with max_depth=30\n",
                "rf_clf = RandomForestClassifier(max_depth=30, n_estimators=100, n_jobs=-1, random_state=42)\n",
                "results(rf_clf)"
            ]
        elif "# TODO: AdaBoostClassifier — try multiple n_estimators values" in source:
            cell['source'] = [
                "# TODO: AdaBoostClassifier — try multiple n_estimators values\n",
                "for n_est in [10, 100, 500]:\n",
                "    print(f'\\nAdaBoostClassifier with n_estimators: {n_est}')\n",
                "    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=n_est, algorithm='SAMME', random_state=42)\n",
                "    results(ada_clf)"
            ]
        elif "# TODO: Scatter plot of X_train, colored by class label (y_train)" in source:
            cell['source'] = [
                "# TODO: Scatter plot of X_train, colored by class label (y_train)\n",
                "plt.figure()\n",
                "plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.7)\n",
                "plt.title('Imbalanced Training Data')\n",
                "plt.show()"
            ]
        elif "# TODO: Fit the classifier on X_train, y_train" in source:
            cell['source'] = [
                "from sklearn.tree import DecisionTreeClassifier\n",
                "\n",
                "# TODO: Fit the classifier on X_train, y_train\n",
                "dt_imba = DecisionTreeClassifier(random_state=42)\n",
                "dt_imba.fit(X_train, y_train)"
            ]
        elif "# TODO: Report accuracy, precision, and recall on the VALIDATION set" in source:
            cell['source'] = [
                "# TODO: Report accuracy, precision, and recall on the VALIDATION set\n",
                "y_val_pred = dt_imba.predict(X_val)\n",
                "\n",
                "print(\"Validation Set:\")\n",
                "print(\"Accuracy: \", dt_imba.score(X_val, y_val))\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score\n",
                "print(\"Precision:\", precision_score(y_val, y_val_pred))\n",
                "print(\"Recall:   \", recall_score(y_val, y_val_pred))"
            ]
        elif "# TODO: Report accuracy, precision, and recall on the TEST set" in source:
            cell['source'] = [
                "# TODO: Report accuracy, precision, and recall on the TEST set\n",
                "y_test_pred = dt_imba.predict(X_test)\n",
                "\n",
                "print(\"Test Set:\")\n",
                "print(\"Accuracy: \", accuracy_score(y_test, y_test_pred))\n",
                "print(\"Precision:\", precision_score(y_test, y_test_pred))\n",
                "print(\"Recall:   \", recall_score(y_test, y_test_pred))"
            ]
        elif "# Store result as X_resampled, y_resampled" in source:
            cell['source'] = [
                "# TODO: Apply RandomUnderSampler to (X_train, y_train)\n",
                "# Store result as X_resampled, y_resampled\n",
                "undersampler = RandomUnderSampler(random_state=42)\n",
                "X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)"
            ]
        elif "# TODO: Scatter plot of resampled data (X_resampled colored by y_resampled)" in source:
            cell['source'] = [
                "# TODO: Scatter plot of resampled data (X_resampled colored by y_resampled)\n",
                "plt.figure()\n",
                "plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, cmap='bwr', alpha=0.7)\n",
                "plt.title('Random Under-Sampled Data')\n",
                "plt.show()"
            ]
        elif "# TODO: Retrain DecisionTreeClassifier on X_resampled, y_resampled" in source:
            cell['source'] = [
                "# TODO: Retrain DecisionTreeClassifier on X_resampled, y_resampled\n",
                "# Report accuracy, precision, and recall on validation AND test sets\n",
                "dt_under = DecisionTreeClassifier(random_state=42)\n",
                "dt_under.fit(X_resampled, y_resampled)\n",
                "\n",
                "y_val_pred_us = dt_under.predict(X_val)\n",
                "y_test_pred_us = dt_under.predict(X_test)\n",
                "\n",
                "print(\"Validation Set:\")\n",
                "print(\"Accuracy: \", accuracy_score(y_val, y_val_pred_us))\n",
                "print(\"Precision:\", precision_score(y_val, y_val_pred_us))\n",
                "print(\"Recall:   \", recall_score(y_val, y_val_pred_us))\n",
                "\n",
                "print(\"\\nTest Set:\")\n",
                "print(\"Accuracy: \", accuracy_score(y_test, y_test_pred_us))\n",
                "print(\"Precision:\", precision_score(y_test, y_test_pred_us))\n",
                "print(\"Recall:   \", recall_score(y_test, y_test_pred_us))"
            ]
        elif "# TODO: Apply SMOTE to (X_train, y_train)" in source:
            cell['source'] = [
                "# TODO: Apply SMOTE to (X_train, y_train)\n",
                "# Retrain DecisionTreeClassifier\n",
                "# Report accuracy, precision, and recall on validation AND test sets\n",
                "smote = SMOTE(random_state=42)\n",
                "X_smote, y_smote = smote.fit_resample(X_train, y_train)\n",
                "\n",
                "dt_smote = DecisionTreeClassifier(random_state=42)\n",
                "dt_smote.fit(X_smote, y_smote)\n",
                "\n",
                "y_val_pred_sm = dt_smote.predict(X_val)\n",
                "y_test_pred_sm = dt_smote.predict(X_test)\n",
                "\n",
                "print(\"Validation Set (SMOTE):\")\n",
                "print(\"Accuracy: \", accuracy_score(y_val, y_val_pred_sm))\n",
                "print(\"Precision:\", precision_score(y_val, y_val_pred_sm))\n",
                "print(\"Recall:   \", recall_score(y_val, y_val_pred_sm))\n",
                "\n",
                "print(\"\\nTest Set (SMOTE):\")\n",
                "print(\"Accuracy: \", accuracy_score(y_test, y_test_pred_sm))\n",
                "print(\"Precision:\", precision_score(y_test, y_test_pred_sm))\n",
                "print(\"Recall:   \", recall_score(y_test, y_test_pred_sm))"
            ]

    elif cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "**Question 1.1:** What happens to the decision boundary as `max_depth` increases?" in source:
            cell['source'] = [
                "**Question 1.1:** What happens to the decision boundary as `max_depth` increases? \n",
                "Is there a risk of overfitting? Explain briefly.\n",
                "\n",
                "*Your answer here:*\n",
                "\n",
                "As `max_depth` increases, the decision boundary becomes more complex and non-linear, fitting tightly to the training points. This increases the risk of overfitting, as the model starts to memorize the noise and specifics of the training data resulting in worse generalization on unseen data.\n"
            ]
        elif "**Question 1.2:** Compare the decision boundaries of BaggingClassifier and RandomForestClassifier." in source:
            cell['source'] = [
                "**Question 1.2:** Compare the decision boundaries of BaggingClassifier and RandomForestClassifier. \n",
                "What is the key difference between Bagging and Random Forest? \n",
                "How does increasing `n_estimators` affect performance and variance?\n",
                "\n",
                "*Your answer here:*\n",
                "\n",
                "Both models present complex boundaries because they aggregate multiple deep trees. However, Random Forest boundaries tend to be smoother and less overfitted to specific instances.\n",
                "\n",
                "The key difference is that Bagging injects randomness only by bootstrapping records, whereas Random Forest also subsets features randomly at each split, further decorrelating the trees.\n",
                "\n",
                "Increasing `n_estimators` directly reduces the variance of the model's predictions because of the averaging, leading to a much more stable and generalized boundary with better test performance, up to a point.\n"
            ]
        elif "**Question 1.3:** How does boosting differ from bagging conceptually?" in source:
            cell['source'] = [
                "**Question 1.3:** How does boosting differ from bagging conceptually? \n",
                "What is the role of the weak learner (`max_depth=1` tree) in AdaBoost?\n",
                "\n",
                "*Your answer here:*\n",
                "\n",
                "Bagging aims to reduce variance by independently training models in parallel and averaging them. Boosting aims to reduce bias by training models sequentially, where each new model attempts to correct the errors of the previous ones.\n",
                "\n",
                "In AdaBoost, the weak learner provides a very simple initial rule slightly better than random guessing. Boosting focuses on the misclassified examples in subsequent iterations, sequentially building on this weak foundation to eventually form a highly accurate strong learner.\n"
            ]
        elif "**Question 2.2:** Is there a gap between precision and recall? Why?" in source:
            cell['source'] = [
                "**Question 2.2:** Is there a gap between precision and recall? Why? \n",
                "Is accuracy a reliable metric here? What does it fail to capture?\n",
                "\n",
                "*Your answer here:*\n",
                "\n",
                "Yes, there is usually a gap. Given the heavy imbalance, the model naturally predicts the majority class to minimize errors. Thus, recall for the minority class drops significantly (it misses actual positives). Precision might stay somewhat decent if the few positives it predicts are accurate, but recall crashes.\n",
                "\n",
                "Accuracy is highly unreliable here. A model could simply predict the majority class for all data and still get a very high accuracy. Accuracy completely fails to capture how poorly the model identifies the minority class.\n"
            ]
        elif "**Question 3.1:** Did precision and recall improve after under-sampling?" in source:
            cell['source'] = [
                "**Question 3.1:** Did precision and recall improve after under-sampling? \n",
                "What is the trade-off when removing majority-class samples?\n",
                "\n",
                "*Your answer here:*\n",
                "\n",
                "Recall usually improves remarkably because the class representation is balanced, making the model more willing to predict the positive class. However, precision can drop, as there are more false positives.\n",
                "\n",
                "The trade-off is information loss. Under-sampling discards potentially useful majority-class data points, leaving fewer examples for the model to effectively learn the true underlying patterns.\n"
            ]
        elif "**Question 3.2:** Compare the three approaches (no resampling, under-sampling, SMOTE)." in source:
            cell['source'] = [
                "**Question 3.2:** Compare the three approaches (no resampling, under-sampling, SMOTE). \n",
                "Which gives the best balance between precision and recall? What are the pros and cons of each method?\n",
                "\n",
                "*Your answer here:*\n",
                "\n",
                "*   **No Resampling**: Gives high precision but terrible recall, keeping high overall accuracy merely because the data is imbalanced. Con: fails on minority class.\n",
                "*   **Under-sampling**: Massively boosts recall but suffers some precision loss. Pro: speeds up training. Con: discards valuable majority-class data.\n",
                "*   **SMOTE**: Consistently achieves the best balance between precision and recall. Pro: adds minority representation without losing majority data. Con: takes longer to process, and could generate synthetic examples in overlapping noisy regions.\n",
                "\n",
                "Overall, SMOTE provides the most balanced and robust performance on minority representations.\n"
            ]

with open('COMP4040_Lab6_Classification_Regression.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
