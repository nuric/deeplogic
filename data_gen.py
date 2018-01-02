"""Data generation script for logic programs."""
import argparse
import random

# Symbol Pool
CONST_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
VAR_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRED_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
EXTRA_SYMBOLS = ".:-,;()"

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
  if length == 1:
    return random.sample(symbols, size)
  rset = set()
  while len(rset) < size:
    rset.add(r_string(symbols, length))
  return list(rset)

def r_consts(size):
  """Return size many unique constants."""
  return r_symbols(size, CONST_SYMBOLS, ARGS.constant_length)

def r_vars(size):
  """Return size many unique variables."""
  return r_symbols(size, VAR_SYMBOLS, ARGS.variable_length)

def r_preds(size):
  """Return size many unique predicates."""
  return r_symbols(size, PRED_SYMBOLS, ARGS.predicate_length)

def write_p(pred):
  """Format single predicate tuple into string."""
  return PRED_T.format(pred[0], ARG_SEP.join(pred[1]))

def write_r(preds):
  """Convert rule predicate tuple into string."""
  head = write_p(preds[0])
  # Is it just a fact
  if len(preds) == 1:
    return FACT_T.format(head)
  # We have a rule
  return RULE_T.format(head, PRED_SEP.join([write_p(p) for p in preds[1:]]))

def output(context, targets):
  """Print the context and given targets."""
  # context: [[('p', ['a', 'b'])], ...]
  # targets: [(('p', ['a', 'b']), 1), ...]
  if ARGS.shuffle_context:
    random.shuffle(context)
  print('\n'.join([write_r(c) for c in context]))
  for t, v in targets:
    print(TARGET_T.format(write_r([t]), v))

def gen_task1(ctx_size):
  """Ground instances only."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  # Create context with both single and double arguments
  ctx, div = list(), ctx_size//2
  for i in range(div):
    args = [random.choice(consts[:-1]), random.choice(consts[:-1])]
    ctx.append([(preds[i], args)])
  for i in range(div, ctx_size):
    ctx.append([(preds[i], [random.choice(consts[:-1])])])
  # Successful case when query appears in context
  targets = [(ctx[0][0], 1), (ctx[div][0], 1)]
  # Out of context constant fails
  if random.random() < 0.5:
    targets.append(((random.choice(preds[:ctx_size//2]), [random.choice(consts), consts[-1]]), 0))
  else:
    targets.append(((random.choice(preds[div:-1]), [consts[-1]]), 0))
  # Out of context predicate fails
  targets.append(((preds[-1], [random.choice(consts[:-1])]), 0))
  output(ctx, targets)

def gen_task2(ctx_size):
  """Variablised facts only."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, div = list(), ctx_size//4
  # Double variable same argument
  for i in range(div):
    v = random.choice(var)
    ctx.append([(preds[i], [v, v])])
  # Double variable unique argument
  for i in range(div, div*2):
    args = random.sample(var, 2)
    ctx.append([(preds[i], args)])
  # Single variable argument
  for i in range(div*2, div*3):
    ctx.append([(preds[i], [random.choice(var)])])
  # Some ground instances
  for i in range(div*3, ctx_size):
    ctx.append([(preds[i], [random.choice(consts)])])
  targets = list()
  # Successful double variable grounding
  p = ctx[div][0][0]
  targets.append(((p, [random.choice(consts), random.choice(consts)]), 1))
  # Successful single variable grounding
  p = ctx[div*2][0][0]
  targets.append(((p, [random.choice(consts)]), 1))
  # Fail on non-unique variable grounding
  p = ctx[0][0][0]
  targets.append(((p, random.sample(consts, 2)), 0))
  # Out of context predicate fails
  targets.append(((preds[-1], [random.choice(consts[:-1])]), 0))
  output(ctx, targets)

def gen_task3(ctx_size):
  """Single step deduction."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, div = list(), ctx_size//2
  # Variable deduction rules
  for i in range(div):
    v = random.choice(var)
    ctx.append([(preds[i*2], [v]), (preds[i*2+1], [v])])
  # Ground instances
  for i in range(div):
    args = random.sample(consts, 2)
    ctx.append([(preds[i*2+1], consts[i])])
  targets = list()
  # Successful deduction
  p = ctx[0][0][0]
  targets.append(((p, [consts[0]]), 1))
  # Successful ground instance
  targets.append((ctx[-1][0], 1))
  # Fail on unknown const deduction
  p = ctx[div-1][0][0]
  targets.append(((p, [random.choice(consts[div:])]), 0))
  # Fail on unsatisfied premise
  p = ctx[1][0][0]
  targets.append(((p, [random.choice(consts[div:])]), 0))
  output(ctx, targets)

if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(description="Generate logic program data.")
  parser.add_argument("task", help="The task to generate.")
  parser.add_argument("size", type=int, help="Number of programs to generate.")
  parser.add_argument("-cs", "--context_size", default=6, type=int, help="Size of program context.")
  parser.add_argument("-cl", "--constant_length", default=1, type=int, help="Length of constants.")
  parser.add_argument("-vl", "--variable_length", default=1, type=int, help="Length of variables.")
  # pylint: disable=line-too-long
  parser.add_argument("-pl", "--predicate_length", default=1, type=int, help="Length of predicates.")
  parser.add_argument("-s", "--shuffle_context", action="store_true", help="Shuffle context before output.")
  ARGS = parser.parse_args()

  # Generate given task
  task = "gen_task" + ARGS.task
  for _ in range(ARGS.size):
    globals()[task](ARGS.context_size)
