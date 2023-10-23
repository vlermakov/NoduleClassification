# Train XG Boost to classify the data
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np

import shap

from sklearn.utils import resample

# Set random seed
#np.random.seed(15)

# Load the data
#data = pd.read_csv('patch_features_new.csv')
data = pd.read_csv('patch_features_older.csv')


# Separate majority and minority classes
df_majority = data[data.label==0]
df_minority = data[data.label==2]

# Find the number of samples needed to upsample the minority class
n_majority = len(df_majority)
n_minority = len(df_minority)
n_samples_to_add = n_majority - n_minority

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=n_samples_to_add    # to match majority class
                                ) # reproducible results

# Combine majority class with upsampled minority class
data = pd.concat([data, df_minority_upsampled])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
data = data.drop([ 'binary_compression','segmentation_type',
                #, 'DOB', 
                #'Age', 
                #     'Gender',
                #   'volume',
                #    #'avg_pixel_value', 
                #    'centroid_x', 'centroid_y',
                #     'centroid_z', 'count', 'depth', 'height', 'multi_component',
                #    # 'position_x', 
                #     'position_y', 
                #    # 'position_z',
                #     'timepoint', 'width',
                #     'world_centroid_x', 'world_centroid_y', 'world_centroid_z',
                #    # 'lld.major.distance', 
                #     'lld.major.p1_x', 'lld.major.p1_y',
                #     'lld.major.p1_z', 'lld.major.p2_x', 'lld.major.p2_y', 'lld.major.p2_z',
                #     'lld.minor.distance', 'lld.minor.p1_x', 'lld.minor.p1_y',
                #     'lld.minor.p1_z', 'lld.minor.p2_x', 'lld.minor.p2_y', 'lld.minor.p2_z',

                 #'Accession_Number',
                  'Phonetic_ID','Patch_ID','id','Slice_Thickness','process', 'thickness','timepoint'], axis=1)

if('Accession_Number' in data.columns):
    data.drop(columns=['Accession_Number'],inplace=True)
# if('Detected_Arterys' in data.columns):
#     data.drop(columns=['Detected_Arterys'],inplace=True)
# if('note' in data.columns):
#     data.drop(columns=['note'],inplace=True)
# if('Location' in data.columns):
#     data.drop(columns=['Location'],inplace=True)
# if('binary' in data.columns):
#     data.drop(columns=['binary'],inplace=True)
# if('X' in data.columns):
#     data.drop(columns=['X'],inplace=True)
# if('Y' in data.columns):
#     data.drop(columns=['Y'],inplace=True)
# if('Z' in data.columns):
#     data.drop(columns=['Z'],inplace=True)
# if('Series' in data.columns):
#     data.drop(columns=['Series'],inplace=True)
# if('Image' in data.columns):
#     data.drop(columns=['Image'],inplace=True)
# if('Size' in data.columns):
#     data.drop(columns=['Size'],inplace=True)
# if('mask_code' in data.columns):
#     data.drop(columns=['mask_code'],inplace=True)
# if('multi_component' in data.columns):
#     data.drop(columns=['multi_component'],inplace=True)
# if('edited' in data.columns):
#     data.drop(columns=['edited'],inplace=True)

# Check if 'Gender' column exists
if('Gender' in data.columns):
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)



# Print stats of number of each class
print(data.label.value_counts())


# # Drop all entries that have class = 2
#data = data[data.label != 1]

# replace label 2 with 1
data['label'] = data['label'].replace(2,1)

# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2)

# Get the indices for training and test sets
for train_idx, test_idx in gss.split(data, groups=data['MRN']):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

# Split the data into features and labels
train_x = train.drop(['label','MRN'],axis=1)
train_y = train['label']

test_x = test.drop(['label','MRN'],axis=1)


test_y = test['label']

# Convert the data into DMatrix format
dtrain = xgb.DMatrix(train_x, label=train_y, enable_categorical=True)
dtest = xgb.DMatrix(test_x, label=test_y, enable_categorical=True)

# Set the parameters for XGBoost
param = {
    #'max_depth': 3,  # the maximum depth of each tree
    #'eta': 0.01,  # the training step for each iteration
    #'objective': 'multi:softprob',  # error evaluation for multiclass training
    "objective": 'binary:logistic',
    #'num_class': 2,  # the number of classes that exist in this datset
    "tree_method": 'gpu_hist',
    'early_stopping_rounds': 10
}

evals = [(dtrain, "train"), (dtest, "validation")]

num_round = 10000  # the number of training iterations

#results = xgb.cv(param, dtrain, num_boost_round = num_round, nfold=5,early_stopping_rounds=20,metrics=["auc"])

#print(results.head())

# Train the model
bst = xgb.train(param, dtrain, num_boost_round = num_round, evals = evals, early_stopping_rounds=20)


preds = bst.predict(dtest)

# Get the predicted labels
#best_preds = np.asarray([np.argmax(line) for line in preds])
best_preds = preds
# Get the accuracy

print("Test Accuracy = {}".format(accuracy_score(test_y, best_preds > 0.5)))
print("Test AUC = {}".format(roc_auc_score(test_y, best_preds)))
print("Test F1 Score = {}".format(f1_score(test_y, best_preds > 0.5)))


explainer = shap.TreeExplainer(bst)
shap_values = explainer(test_x)

# Summary plot
shap.summary_plot(shap_values, test_x)

# Individual SHAP value plot (for the first instance in the test set)
#shap.plots.waterfall(shap_values[0], max_display=20)

#shap.plots.bar(shap_values)
shap.plots.scatter(shap_values[:,"lld.major.distance"], color=shap_values[:, "avg_pixel_value"])

# # Plot the ROC curve with the test results
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# fpr, tpr, _ = roc_curve(test_y, preds)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#          lw=2, label='ROC curve (area = {})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('1 - Specificity')
# plt.ylabel('Sensitivity')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()


# Print Train metrics
preds = bst.predict(dtrain)
#best_preds = np.asarray([np.argmax(line) for line in preds])
best_preds = preds
print("Train Accuracy = {}".format(accuracy_score(train_y, best_preds > 0.5)))
print("Train AUC = {}".format(roc_auc_score(train_y, best_preds)))
print("Train F1 Score = {}".format(f1_score(train_y, best_preds > 0.5)))


# Perform 5 fold cross validation
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Initialize GroupKFol=


# Get XGBoost feature importance
import matplotlib.pyplot as plt

xgb.plot_importance(bst,importance_type='weight')
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
