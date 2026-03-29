from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from src.params import dt_params, svc_params, lr_params, vc_params, SEED

def model():
  dt_clf = DecisionTreeClassifier(random_state=SEED, **dt_params)
  svc_clf = SVC(random_state=SEED, probability=True, **svc_params)
  lr_clf = LogisticRegression(random_state=SEED, **lr_params)

  voting_clf = VotingClassifier(
    estimators=[
      ("dt", dt_clf),
      ("svc", svc_clf),
      ("lr", lr_clf)
    ],
    **vc_params
  )
  
  return voting_clf