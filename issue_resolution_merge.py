import resolutionEvalSingle
import issueEvalSingle
from resolutionEvalSingle import eval
from issueEvalSingle import eval
import sys

def predict_issue_resolution():
    x_raw = sys.argv[1]
    checkpoint_dir_issue = sys.argv[2]
    checkpoint_dir_resolution = sys.argv[3]
    issue = issueEvalSingle.eval(x_raw,checkpoint_dir_issue)
    resolution = resolutionEvalSingle.eval(x_raw,checkpoint_dir_resolution)
    issue.update(resolution)
    return issue

print(predict_issue_resolution())
