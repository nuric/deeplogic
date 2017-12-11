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

def writep(pred):
  """Format single predicate tuple into string."""
  return PRED_T.format(pred[0], ARG_SEP.join(pred[1]))

def write(preds, isquery=False):
  """Convert single predicate tuple into string."""
  head = writep(preds[0])
  # Is it just a fact
  if len(preds) == 1:
    return head if isquery else FACT_T.format(head)
  # We have a rule
  return RULE_T.format(head, PRED_SEP.join([writep(p) for p in preds[1:]]))

def gen_task1(context_size):
  """Ground instances only."""
  preds = r_preds(context_size+1)
  consts = r_consts(context_size+1)
  # Create context with both single and double arguments
  ctx = list()
  for i in range(context_size//2):
    args = [random.choice(consts[:-1]), random.choice(consts[:-1])]
    ctx.append([(preds[i], args)])
  for i in range(context_size//2, context_size):
    ctx.append([(preds[i], [random.choice(consts[:-1])])])
  random.shuffle(ctx)
  print('\n'.join([write(c) for c in ctx]))
  # Successful case when query appears in context
  target_ts = random.sample(ctx, 2)
  for t in target_ts:
    print(TARGET_T.format(write(t, True), 1))
  # Out of context constant fails
  target_f = (random.choice(preds[context_size//2:-1]), [consts[-1]])
  print(TARGET_T.format(writep(target_f), 0))
  # Out of context predicate fails
  target_f = (preds[-1], [random.choice(consts[:-1])])
  print(TARGET_T.format(writep(target_f), 0))

if __name__ == '__main__':
  task = "gen_task" + ARGS.task
  for _ in range(ARGS.size):
    globals()[task](ARGS.context_size)
