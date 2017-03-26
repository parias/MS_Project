#!/bin/bash
for j in $(seq 100 5 140); # batch_size
   do for k in $(seq 22 2 30); # num_hidden
      do python3 rnn.py 5000 $j $k; #rnn(epoch, batch_size num_hidden)
   done;
done;
