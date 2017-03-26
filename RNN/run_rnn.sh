#!/bin/bash
for i in $(seq 5000 1000 7000); # epoch
   do for j in $(seq 100 100 500); # batch_size
      do for k in $(seq 22 2 30); # num_hidden
         do python3 rnn.py $i $j $k; #rnn(epoch, batch_size num_hidden)
      done;
   done;
done;
