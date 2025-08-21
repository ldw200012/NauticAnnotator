#!/usr/bin/python3

import rospy
import sys
import time

def main():
    rospy.init_node('nodes_processtime_handler')
    frame_num = 10
    rate = rospy.Rate(frame_num) # 10 Hz

    dot_cnt_max = 50
    dot_cnt_div = frame_num
    dot_cnt = 1

    while not rospy.is_shutdown():
        node1_processtime = rospy.get_param("node1_processtime", 0.00)
        node2_processtime = rospy.get_param("node2_processtime", 0.00)

        # Move the cursor up N lines
        sys.stdout.write('\x1b[2A')

        # Node 1
        sys.stdout.write('\x1b[2K')
        node1_timeprint = f"{node1_processtime:.2f}".zfill(6)
        print(f"[nautic_annotator] ==> [detection.py] node \033[91mRUNNING [{node1_timeprint}]ms\033[0m " + "." * (int(dot_cnt//dot_cnt_div)), flush=True)

        # Node 2
        sys.stdout.write('\x1b[2K')
        node2_timeprint = f"{node2_processtime:.2f}".zfill(6)
        print(f"[nautic_annotator] ==> [nautic_annotator] node \033[91mRUNNING [{node2_timeprint}]ms\033[0m " + "." * (int(dot_cnt//dot_cnt_div)), flush=True)

        dot_cnt = (dot_cnt+1) if dot_cnt < dot_cnt_max else 1

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass