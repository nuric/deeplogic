"""Data generation script for logic programs."""
import argparse
import random as R

# Symbol Pool
CONST_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
VAR_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRED_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
EXTRA_SYMBOLS = ".:-,;()"

CHARS = sorted(list(set(CONST_SYMBOLS+VAR_SYMBOLS+PRED_SYMBOLS+EXTRA_SYMBOLS)))
# Reserve 0 for padding
CHAR_IDX = dict((c, i+1) for i, c in enumerate(CHARS))
IDX_CHAR = [0]
IDX_CHAR.extend(CHARS)

# Predicate Templates
FACT_T = "{}."
RULE_T = "{}:-{}."
PRED_T = "{}({})"
ARG_SEP = ','
PRED_SEP = ';'
TARGET_T = "? {} {}"

def r_string(symbols, length):
  """Return random sequence from given symbols."""
  return ''.join(R.choice(symbols)
                 for _ in range(length))

def r_symbols(size, symbols, length):
  """Return unique random from given symbols."""
  if length == 1:
    return R.sample(symbols, size)
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
    R.shuffle(context)
  print('\n'.join([write_r(c) for c in context]))
  for t, v in targets:
    print(TARGET_T.format(write_r([t]), v))

def gen_task1(ctx_size):
  """Ground instances only: p(a).q(c,b)."""
  assert ctx_size >= 1
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  # Create context with both single and double arguments
  ctx = list()
  for i in range(ctx_size):
    if R.random() < 0.5:
      args = [R.choice(consts[:-1]), R.choice(consts[:-1])]
    else:
      args = [R.choice(consts[:-1])]
    ctx.append([(preds[i], args)])
  # Successful case when query appears in context
  targets = [(ctx[0][0], 1)]
  # What are possible failures
  fails = list()
  # Out of context constant fails
  pred = R.choice(ctx)[0]
  args = pred[1].copy()
  args[R.randrange(len(args))] = consts[-1]
  fails.append(((pred[0], args), 0))
  # Out of context predicate fails
  fails.append(((preds[-1], [R.choice(consts[:-1])]), 0))
  targets.append(R.choice(fails))
  output(ctx, targets)

def gen_task2(ctx_size):
  """Variablised facts only: p(X).q(X,Y)."""
  assert ctx_size >= 1
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  for i in range(ctx_size):
    rtype = R.randrange(4)
    if rtype == 0:
      # Double variable same argument
      v = R.choice(var)
      ctx.append([(preds[i], [v, v])])
      if i == 0:
        # Successful double variable grounding
        c = R.choice(consts)
        targets.append(((preds[i], [c, c]), 1))
        # Fail on non-unique variable grounding
        targets.append(((preds[i], R.sample(consts, 2)), 0))
    elif rtype == 1:
      # Double variable unique argument
      args = R.sample(var, 2)
      ctx.append([(preds[i], args)])
      if i == 0:
        # Successful unique argument grounding
        args = [R.choice(consts), R.choice(consts)]
        targets.append(((preds[i], args), 1))
        # Fail on out of context predicate with same arguments
        targets.append(((preds[-1], args), 0))
    elif rtype == 2:
      # Single variable argument
      ctx.append([(preds[i], [R.choice(var)])])
      if i == 0:
        # Successful argument grounding
        args = [R.choice(consts)]
        targets.append(((preds[i], args), 1))
        # Fail on out of context predicate
        targets.append(((preds[-1], args), 0))
    else:
      # Some ground instances
      if R.random() < 0.5:
        args = [R.choice(consts[:-1]), R.choice(consts[:-1])]
      else:
        args = [R.choice(consts[:-1])]
      pred = (preds[i], args)
      ctx.append([pred])
      if i == 0:
        # This is same as task 1 (?)
        # Successful ground case
        targets.append((pred, 1))
        # Fail on different constants
        args = pred[1].copy()
        args[R.randrange(len(args))] = consts[-1]
        targets.append(((pred[0], args), 0))
  output(ctx, targets)

def nstep_deduction(ctx_size, steps):
  assert steps >= 1
  assert ctx_size >= (steps + 1)
  preds = r_preds(ctx_size*2+steps)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  i, pidx = 0, 0
  while i < ctx_size:
    rtype = R.randrange(3 if i == 0 else 4)
    if rtype == 0:
      # Double variable swap deduction rules
      vs = R.sample(var, 2)
      ctx.append([(preds[pidx], vs), (preds[pidx+1], vs[::-1])])
      if i == 0:
        # Add the n steps
        for j in range(steps-1):
          vs = R.sample(var, 2)
          ctx.append([(preds[pidx+j+1], vs), (preds[pidx+j+2], vs[::-1])])
        # Add the ground case
        args = R.sample(consts[:-1], 2)
        ctx.append([(preds[pidx+steps], args)])
        i += steps
        targets.append(((preds[pidx], args), 1))
        targets.append(((preds[pidx], args[::-1]), 0))
        pidx += steps-1
      pidx += 2
    elif rtype == 1:
      # Double variable non-swap deduction rules
      vs = R.sample(var, 2)
      ctx.append([(preds[pidx], vs), (preds[pidx+1], vs)])
      if i == 0:
        # Add the n steps
        for j in range(steps-1):
          vs = R.sample(var, 2)
          ctx.append([(preds[pidx+j+1], vs), (preds[pidx+j+2], vs)])
        args = [R.choice(consts[:-1]), R.choice(consts[:-1])]
        # Add the ground case
        ctx.append([(preds[pidx+steps], args)])
        i += steps
        targets.append(((preds[pidx], args), 1))
        # Fail on either missing premise or constant
        if R.random() < 0.5:
          targets.append(((preds[-1], args), 0))
        else:
          args = args.copy()
          args[R.randrange(len(args))] = consts[-1]
          targets.append(((preds[pidx], args), 0))
        pidx += steps-1
      pidx += 2
    elif rtype == 2:
      # Single variable deduction rules
      v = R.choice(var)
      ctx.append([(preds[pidx], [v]), (preds[pidx+1], [v])])
      if i == 0:
        # Add the n steps
        for j in range(steps-1):
          v = R.choice(var)
          ctx.append([(preds[pidx+j+1], [v]), (preds[pidx+j+2], [v])])
        # Add the ground case
        c = R.choice(consts[:-1])
        ctx.append([(preds[pidx+steps], [c])])
        i += steps
        targets.append(((preds[pidx], [c]), 1))
        # Fail on either missing premise or constant
        if R.random() < 0.5:
          targets.append(((preds[-1], [c]), 0))
        else:
          targets.append(((preds[pidx], [consts[-1]]), 0))
        pidx += steps-1
      pidx += 2
    else:
      # Ground instances
      ctx.append([(preds[pidx], [R.choice(consts[:-1])])])
      pidx += 1
    i += 1
  output(ctx, targets)

def gen_task3(ctx_size):
  """Single step deduction: p(X):-q(X)."""
  nstep_deduction(ctx_size, 1)

def gen_task4(ctx_size):
  """Double step deduction: p(X):-q(X).q(X):-r(X)."""
  nstep_deduction(ctx_size, 2)

def gen_task5(ctx_size):
  """Triple step deduction."""
  nstep_deduction(ctx_size, 3)

def gen_task6(ctx_size):
  """Logical AND: p(X):-q(X);r(X)."""
  assert ctx_size >= 3
  preds = r_preds(ctx_size*3+1)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  i, pidx = 0, 0
  while i < ctx_size:
    rtype = R.randrange(2 if i == 0 else 3)
    if rtype == 0:
      # Double variable AND with different vars
      vs = R.sample(var, 2)
      ctx.append([(preds[pidx], vs),
                  (preds[pidx+1], vs[:1]),
                  (preds[pidx+2], vs[1:])])
      if i == 0:
        # Add the ground cases
        args = [R.choice(consts[:-1]), R.choice(consts[:-1])]
        ctx.append([(preds[pidx+1], args[:1])])
        ctx.append([(preds[pidx+2], args[1:])])
        i += 2
        # Successful case
        targets.append(((preds[pidx], args), 1))
        # Fail on non-matching constant
        args = args.copy()
        args[R.randrange(len(args))] = consts[-1]
        targets.append(((preds[pidx], args), 0))
      pidx += 3
    elif rtype == 1:
      # Single variable AND
      v = R.choice(var)
      ctx.append([(preds[pidx], [v]),
                  (preds[pidx+1], [v]),
                  (preds[pidx+2], [v])])
      if i == 0:
        # Add the ground cases
        c = R.choice(consts[:-1])
        ctx.append([(preds[pidx+1], [c])])
        ctx.append([(preds[pidx+2], [c])])
        i += 2
        targets.append(((preds[pidx], [c]), 1))
        targets.append(((preds[pidx], [consts[-1]]), 0))
      pidx += 3
    else:
      # Some other ground cases
      if R.random() < 0.5:
        args = [R.choice(consts), R.choice(consts)]
      else:
        args = [R.choice(consts)]
      ctx.append([(preds[pidx], args)])
      pidx += 1
    i += 1
  output(ctx, targets)

def gen_task7(ctx_size):
  """Transitive case: p(X,Y):-q(X,Z);R(Z,Y)."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, div = list(), ctx_size//3
  # Transitive rules
  for i in range(div):
    vs = R.sample(var, 3)
    r = [(preds[i*3], [vs[0], vs[2]]),
         (preds[i*3+1], [vs[0], vs[1]]),
         (preds[i*3+2], [vs[1], vs[2]])]
    ctx.append(r)
  # Ground instances to satisfy deduction
  for i in range(div//2):
    ctx.append([(preds[i*3+1], [consts[i*2], consts[i*2+1]])])
    ctx.append([(preds[i*3+2], [consts[i*2+1], consts[i*2+2]])])
  for i in range(div//2, div):
    ctx.append([(preds[i*3+1], [consts[i*2], consts[i*2+1]])])
    ctx.append([(preds[i*3+2], [consts[-1], consts[i*2+2]])])
  targets = list()
  # Successful deduction
  p = ctx[0][0][0]
  targets.append(((p, [consts[0], consts[2]]), 1))
  # Fail on non-matching premise
  p = ctx[div//2][0][0]
  targets.append(((p, [consts[div], consts[div+2]]), 0))
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
