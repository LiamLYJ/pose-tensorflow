import os
from subprocess import check_output

command = 'python -u demo/get_logistic_14.py 2>&1 | tee -a log.txt'
# os.system(command)
PID = str(check_output(["pidof","python"]))
print (PID)
if PID.count(' ') >= 2:
    print ('its runing something already~~')
else:
    print ('process doesnot exist, start to launch')
    os.system(command)
