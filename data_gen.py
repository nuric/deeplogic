"""Data generation script for logic programs."""
import argparse
import random

# Arguments
parser = argparse.ArgumentParser(description="Generate logic program data.")
parser.add_argument("task", help="The task to generate.")
parser.add_argument("size", type=int, help="Number of programs to generate.")
parser.add_argument("-cs", "--context_size", default=4, type=int, help="Size of program context.")
parser.add_argument("-cl", "--constant_length", default=1, type=int, help="Length of constants.")
parser.add_argument("-vl", "--variable_length", default=1, type=int, help="Length of variables.")
parser.add_argument("-pl", "--predicate_length", default=1, type=int, help="Length of predicates.")
ARGS = parser.parse_args()

# Symbol Pool
CONST_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
VAR_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRED_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"

# Predicate Templates
FACT_T = "{}."
RULE_T = "{}:-{}."
PRED_T = "{}({})"
ARG_SEP = ','
PRED_SEP = ';'
TARGET_T = "? {} {}"

def r_string(symbols, length):
  """Return random sequence from given symbols."""
  return ''.join(random.choice(symbols)
                 for _ in range(length))

def r_symbols(size, symbols, length):
  """Return unique random from given symbols."""
  r = set()
  while len(r) < size:
    r.add(r_string(symbols, length))
  return list(r)

def r_consts(size):
  """Return size many unique constants."""
  return r_symbols(size, CONST_SYMBOLS, ARGS.constant_length)

def r_vars(size):
  """Return size many unique variables."""
  return r_symbols(size, VAR_SYMBOLS, ARGS.variable_length)

def r_preds(size):
  """Return size many unique predicates."""
  return r_symbols(size, PRED_SYMBOLS, ARGS.predicate_length)

def gen_task1(context_size):
  """Ground instances only."""
  preds = zip(r_preds(context_size+1), r_consts(context_size+1))
  preds = [PRED_T.format(*pc) for pc in preds]
  ctx = [FACT_T.format(c) for c in preds[:-1]]
  target_t, target_f = preds[0], preds[-1]
  random.shuffle(ctx)
  print('\n'.join(ctx))
  print(TARGET_T.format(target_t, 1))
  print(TARGET_T.format(target_f, 0))

if __name__ == '__main__':
  task = "gen_task" + ARGS.task
  for _ in range(ARGS.size):
    globals()[task](ARGS.context_size)
