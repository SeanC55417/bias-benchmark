from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('aif/mimic_cases (1).csv')

outcome = 'observed_admit'

# drop missing values 
dataset = dataset.dropna()

#make binary columns
dataset['race_binary'] = dataset['race'].apply(lambda x: 1 if x == 'WHITE' else 0)
dataset['housing_binary'] = dataset['housing_stability_proxy'].apply(lambda x: 1 if x == 'stable' else 0)
dataset['gender_binary'] = dataset['gender'].apply(lambda x: 1 if x == 'M' else 0)
dataset['caregiver_binary'] = dataset['caregiver_present'].apply(lambda x: 1 if x == 'yes' else 0)
dataset['insurance_binary'] = dataset['coverage_proxy'].apply(lambda x: 1 if x == 'private' else 0)

fairness_features = [
    ('Race', 'race_binary'),
    ('Housing Stability', 'housing_binary'),
    ('Gender', 'gender_binary'),
    ('Caregiver Presence', 'caregiver_binary'),
    ('Insurance Status', 'insurance_binary')
]


for name, col in fairness_features:

    aif_dataset = BinaryLabelDataset(
        df=dataset[[col, outcome]].astype(float),
        label_names=[outcome],
        protected_attribute_names=[col]
    )

    privileged_groups = [{col: 1}]
    unprivileged_groups = [{col: 0}]

    metric = BinaryLabelDatasetMetric(
        aif_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    print(f"Disparate Impact in terms of {name}: {metric.disparate_impact()}")

#VISUALS

categories = ["Race", "Housing Stability", "Gender", "Caregiver Presence", "Insurance Status"]
values = [0.7543, 0.6980, 0.7418, 0.9302,0.7575]

plt.figure()
plt.barh(categories, values)

#this is the fairness threshold, anything below indicates adverse impact on underprivileged group, 1.0 is perfect fairness
plt.axvline(x=0.8)

plt.xlabel("Disparate Impact")
plt.title("Disparate Impact by Category")

plt.show()

#RATIO
comparisons = [
    "Female vs Male",
    "White vs Non-White",
    "Insurance: Non-Private vs Private",
    "Unstable vs Stable Housing",
    "Caregiver Presence: Yes vs No"
]

values = [0.887978, 0.9787, 1.023810, 0.882927, 1.039185]

plt.figure()
plt.barh(comparisons, values)

# threshold line
plt.axvline(x=0.8) #<0.8Potential disadvantage to the unprivileged group.
plt.axvline(x=1.25) #> 1.25 Potential unfair advantage to the unprivileged group


plt.xlabel("Disparity Ratio")
plt.title("Disparity Ratios by Group Comparison")

plt.show()
