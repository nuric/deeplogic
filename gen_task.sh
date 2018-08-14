#!/bin/bash
# This script wraps to generate programs with varying parameters
FUNC=$1
DDIR='data/'
DCMD='python3 data_gen.py'
shift
SIZE=$1
TSIZE=$((SIZE / 10))
shift
ARGS='-pl 2 -cl 2 -ns 2'

eval_single() {
  echo "Generating evaluation data for all tasks..."
  for i in {1..12}; do
    F=$DDIR'test_task'$i.txt
    echo Writing to $F
    $DCMD $ARGS -t $i -s $SIZE > $F
  done
}

eval_nstep() {
  echo "Generating nstep evaluation data."
  rm -f $DDIR'test_nstep'*.txt
  for i in {1..24}; do
    F=$DDIR'test_nstep'$i.txt
    echo Writing to $F
    $DCMD $ARGS -s $SIZE --nstep $i > $F
  done
}

eval_len() {
  echo "Generating increasing length data."
  rm -f $DDIR'test_pl'*.txt
  rm -f $DDIR'test_cl'*.txt
  for i in {2..64}; do
    F=$DDIR'test_pl'$i.txt
    echo Writing to $F
    $DCMD -s $SIZE -t 3 -pl $i -cl 2 > $F
    F=$DDIR'test_cl'$i.txt
    echo Writing to $F
    $DCMD -s $SIZE -t 3 -pl 2 -cl $i > $F
  done
}

acc() {
  echo "Generating accumulating training data..."
  for i in {1..5}; do
    F=$DDIR'train_task1-'$i.txt
    TF=$DDIR'test_task1-'$i.txt
    echo Writing to $F $TF
    rm -f $F $TF
    for j in $(seq $i); do
      $DCMD $ARGS -t $j -s $SIZE >> $F
      $DCMD $ARGS -t $j -s $TSIZE >> $TF
    done
  done
}

iter() {
  echo "Generating tasks based on iteration..."
  TS[1]="1 2"
  TS[2]="1 2 3 7 9 12"
  TS[3]="1 2 3 4 6 7 8 9 10 11 12"
  for i in {1..3}; do
    F=$DDIR'train_iter'$i.txt
    TF=$DDIR'test_iter'$i.txt
    echo Writing to $F $TF
    rm -rf $F $TF
    for j in ${TS[i]}; do
      $DCMD $ARGS -t $j -s $SIZE >> $F
      $DCMD $ARGS -t $j -s $TSIZE >> $TF
    done
  done
}

all() {
  echo "Generating all tasks..."
  for i in {1..12}; do
    F=$DDIR'train_task'$i.txt
    TF=$DDIR'val_task'$i.txt
    echo Writing to $F $TF
    rm -f $F $TF
    $DCMD $ARGS -t $i -s $SIZE >> $F
    $DCMD $ARGS -t $i -s $TSIZE >> $TF
  done
}

custom() {
  echo "Generating certain tasks only..."
  F=$DDIR'train_custom.txt'
  TF=$DDIR'test_custom.txt'
  echo Writing to $F $TF
  rm -f $F $TF
  for i in "$@"; do
    echo Generating task $i
    $DCMD $ARGS -t $i -s $SIZE >> $F
    $DCMD $ARGS -t $i -s $TSIZE >> $TF
  done
}

# Run given function
$FUNC "$@"
