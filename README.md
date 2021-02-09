# Sk-LMN
Mass-based Similarity Weighted k-Neighbors for Class Imbalance

Parameters:

-data: Data input file, must be placed under "data" folder : (default="01--glass1.csv")

-n_estimators: Set number of IsoationForest trees : (default =1)

-max_samples: Set number of samples on each IsoationForest tree : (default =-1)

-k: kNN (default=3)

-split_seed: Seed for spliting traing and testing sets (default=42)

-test_ratio: Fragemnt of test set (default=0.2)

Example: 

python SkLMN.py -data  01--glass1.csv -k 10

Results will be written to the csv file RESULT/SkLMN.csv

Reference: *To be updated soon
