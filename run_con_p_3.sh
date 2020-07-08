ps aux|grep preduce|grep -v grep|cut -c 9-15|xargs kill -9
python preduce_con_controller.py -r 0 -s 9 -k 3 &
python preduce_con_worker.py -s 9 -r 1 -g 0 -k 3 &
python preduce_con_worker.py -s 9 -r 2 -g 1 -k 3 &
python preduce_con_worker.py -s 9 -r 3 -g 2 -k 3 &
python preduce_con_worker.py -s 9 -r 4 -g 3 -k 3 &
python preduce_con_worker.py -s 9 -r 5 -g 4 -k 3 &
python preduce_con_worker.py -s 9 -r 6 -g 5 -k 3 &
python preduce_con_worker.py -s 9 -r 7 -g 6 -k 3 &
python preduce_con_worker.py -s 9 -r 8 -g 7 -k 3
