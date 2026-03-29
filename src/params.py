SEED = 42

dt_params = {
  'criterion': 'entropy', 
  'max_depth': 7, 
  'max_features': 'sqrt', 
  'max_leaf_nodes': None, 
  'min_samples_leaf': 1, 
  'min_samples_split': 2
}

svc_params = {
  'C': 1, 
  'break_ties': True, 
  'class_weight': None, 
  'decision_function_shape': 'ovr', 
  'degree': 2, 
  'gamma': 'auto', 
  'kernel': 'rbf', 
  'max_iter': 200, 
  'shrinking': True
}

lr_params = {
  'C': 0.01, 
  'max_iter': 100, 
  'penalty': 'l2', 
  'solver': 'liblinear', 
  'warm_start': True
}

vc_params = {
  'flatten_transform': True, 
  'voting': 'hard', 
  'weights': None
}
