#!/usr/bin/python3

import os

rows_list = os.listdir("tuning")
for rows in rows_list:
  cols_list = os.listdir("tuning/"+rows)
  for cols in cols_list:
    with open("tuning/"+rows+"/"+cols+"/"+"main.stdout") as fp:
      lines = fp.readlines()
      cuTimes = [int(val) for val in lines[0].strip().split(" ")]
      myTimes = [int(val) for val in lines[1].strip().split(" ")]
      cuAvg = 0.0
      cuDev = 0.0
      myAvg = 0.0
      myDev = 0.0
      for t in cuTimes:
        cuAvg += t
      cuAvg /= len(cuTimes)
      for t in myTimes:
        myAvg += t
      myAvg /= len(myTimes)
      for t in cuTimes:
        cuDev += (t - cuAvg)**2
      cuDev = (cuDev/len(cuTimes)/(len(cuTimes)-1))**(1/2)
      for t in myTimes:
        myDev += (t - myAvg)**2
      myDev = (myDev/len(myTimes)/(len(myTimes)-1))**(1/2)
      # print(rows, cols, cuAvg, cuDev, myAvg, myDev)
      cuRel = cuDev/cuAvg
      myRel = myDev/myAvg
      totRel = (cuRel**2 + myRel**2)**(1/2)
      print(rows, cols, cuAvg/myAvg, totRel*(cuAvg/myAvg))