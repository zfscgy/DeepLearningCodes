'''
    This file is use to run task on ubuntu server conveniently
    Usage:
        python(3) RunTask.py dcgan
'''
import sys
sys.path.append("/home/xly/zf/DeepLearning/DeepLearning")
import os
os.chdir("/home/xly/zf/DeepLearning/DeepLearning")

task_dict = {
    "dcgan": "Tasks/ImageGeneration/DCGAN.py",
    "nextitnet": "Tasks/SessionRecommendation/SimpleCOnvNetForNextItem.py",
    "gru4rec": "Tasks/SessionRecommendation/Gru4Rec.py",
    "oneshot": "Tasks/TransferLearning/OneShotLearning"
}
if len(sys.argv) < 2:
    print("You can run following tasks")
    for key in task_dict:
        print(key)
    exit()
task = sys.argv[1]
if task not in task_dict:
    print("Task ", task, " not found")
    for key in task_dict:
        print(key)
    exit()
else:
    exec(open(task_dict[sys.argv[1]]).read())
