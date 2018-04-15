#!/bin/bash
# This script wraps to generate programs with varying parameters
FUNC=$1
DDIR='data/'
DCMD='python3 data_gen.py'
SIZE=$2
TSIZE=$((SIZE / 10))

# Task 1: ground instances
gen_task1() {
  $DCMD -cs 2 -cl 1 -pl 1 1 $1
  $DCMD -cs 4 -cl 2 -pl 2 1 $1
  $DCMD -cs 4 -cl 2 -pl 1 1 $1
  $DCMD -cs 2 -cl 1 -pl 2 1 $1
}

# Task 2: variable binding
gen_task2() {
  $DCMD -cs 4 -cl 1 -pl 1 2 $1
  $DCMD -cs 2 -cl 2 -pl 1 2 $1
  $DCMD -cs 2 -cl 2 -pl 2 2 $1
  $DCMD -cs 4 -cl 1 -pl 2 2 $1
}

# Task 3: single step deduction
gen_task3() {
  $DCMD -cs 4 -cl 1 -pl 1 3 $1
  $DCMD -cs 5 -cl 1 -pl 2 3 $1
  $DCMD -cs 4 -cl 2 -pl 2 3 $1
  $DCMD -cs 5 -cl 2 -pl 1 3 $1
}

# Task 4: double step deduction
gen_task4() {
  $DCMD -cs 6 -cl 1 -pl 1 4 $1
  $DCMD -cs 5 -cl 2 -pl 2 4 $1
  $DCMD -cs 5 -cl 2 -pl 1 4 $1
  $DCMD -cs 6 -cl 1 -pl 2 4 $1
}

# Task 5: triple step deduction
gen_task5() {
  $DCMD -cs 7 -cl 1 -pl 1 5 $1
  $DCMD -cs 6 -cl 2 -pl 2 5 $1
  $DCMD -cs 6 -cl 2 -pl 1 5 $1
  $DCMD -cs 7 -cl 1 -pl 2 5 $1
}

# Task 6: logical and
gen_task6() {
  $DCMD -cs 5 -cl 1 -pl 1 6 $1
  $DCMD -cs 4 -cl 2 -pl 2 6 $1
  $DCMD -cs 4 -cl 2 -pl 1 6 $1
  $DCMD -cs 5 -cl 1 -pl 2 6 $1
}

# Task 7: logical or
gen_task7() {
  $DCMD -cs 4 -cl 1 -pl 1 7 $1
  $DCMD -cs 5 -cl 2 -pl 2 7 $1
  $DCMD -cs 4 -cl 2 -pl 1 7 $1
  $DCMD -cs 5 -cl 1 -pl 2 7 $1
}

# Task 8: transitive case
gen_task8() {
  $DCMD -cs 6 -cl 1 -pl 1 8 $1
  $DCMD -cs 5 -cl 2 -pl 2 8 $1
  $DCMD -cs 5 -cl 2 -pl 1 8 $1
  $DCMD -cs 6 -cl 1 -pl 2 8 $1
}

# Task 9: single step deduction with NBF
gen_task9() {
  $DCMD -cs 5 -cl 1 -pl 1 9 $1
  $DCMD -cs 4 -cl 2 -pl 2 9 $1
  $DCMD -cs 4 -cl 2 -pl 1 9 $1
  $DCMD -cs 5 -cl 1 -pl 2 9 $1
}

# Task 10: double step deduction with NBF
gen_task10() {
  $DCMD -cs 5 -cl 1 -pl 1 10 $1
  $DCMD -cs 6 -cl 2 -pl 2 10 $1
  $DCMD -cs 6 -cl 2 -pl 1 10 $1
  $DCMD -cs 5 -cl 1 -pl 2 10 $1
}

# Task 11: logical and with NBF
gen_task11() {
  $DCMD -cs 5 -cl 1 -pl 1 11 $1
  $DCMD -cs 6 -cl 2 -pl 2 11 $1
  $DCMD -cs 5 -cl 2 -pl 1 11 $1
  $DCMD -cs 6 -cl 1 -pl 2 11 $1
}

# Task 12: logical or with NBF
gen_task12() {
  $DCMD -cs 5 -cl 1 -pl 1 12 $1
  $DCMD -cs 4 -cl 2 -pl 2 12 $1
  $DCMD -cs 4 -cl 2 -pl 1 12 $1
  $DCMD -cs 5 -cl 1 -pl 2 12 $1
}

eval_single() {
  echo "Generating evaluation data for all tasks..."
  for i in {1..12}; do
    F=$DDIR'test_task'$i.txt
    echo Writing to $F
    gen_task$i $SIZE > $F
  done
}

eval_nstep() {
  echo "Generating nstep evaluation data."
  for i in {2..38..4}; do
    F=$DDIR'test_nstep'$i.txt
    echo Writing to $F
    $DCMD -s $SIZE -ns $i -cs $((i + 4)) -pl 2 -cl 2 > $F
  done
}

acc() {
  echo "Generating accumulating training data..."
  for i in {1..12}; do
    F=$DDIR'train_task1-'$i.txt
    TF=$DDIR'test_task1-'$i.txt
    echo Writing to $F $TF
    rm -f $F $TF
    for j in $(seq $i); do
      gen_task$j $SIZE >> $F
      gen_task$j $TSIZE >> $TF
    done
  done
}

all() {
  echo "Generating all tasks..."
  F=$DDIR'train.txt'
  TF=$DDIR'test.txt'
  echo Writing to $F $TF
  rm -f $F $TF
  for i in {1..12}; do
    gen_task$i $SIZE >> $F
    gen_task$i $TSIZE >> $TF
  done
}

custom() {
  echo "Generating certain tasks only..."
  F=$DDIR'train.txt'
  TF=$DDIR'test.txt'
  echo Writing to $F $TF
  rm -f $F $TF
  for i in 1 2 3 4 6 8 9 10 11; do
    gen_task$i $SIZE >> $F
    gen_task$i $TSIZE >> $TF
  done
}

# Run given function
$FUNC
