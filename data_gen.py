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

def choices(symbols, k):
  """Return k many symbols with replacement. Added in v3.6."""
  return [R.choice(symbols) for _ in range(k)]

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
    k = 2 if R.random() < 0.5 else 1
    ctx.append([(preds[i], choices(consts[:-1], k))])
  # Successful case when query appears in context
  targets = [(ctx[0][0], 1)]
  # What are possible failures
  if R.random() < 0.5:
    # Out of context constant fails
    pred = R.choice(ctx)[0]
    args = pred[1].copy()
    args[R.randrange(len(args))] = consts[-1]
    targets.append(((pred[0], args), 0))
  else:
    # Out of context predicate fails
    targets.append(((preds[-1], choices(consts[:-1], 1)), 0))
  output(ctx, targets)

def gen_task2(ctx_size):
  """Variablised facts only: p(X).q(X,Y)."""
  assert ctx_size >= 1
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  for i in range(ctx_size):
    rtype = R.randrange(2 if i == 0 else 3)
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
      # Double variable different argument
      # Single variable argument
      argc = R.randint(1, 2)
      args = R.sample(var, argc)
      ctx.append([(preds[i], args)])
      if i == 0:
        # Successful unique argument grounding
        args = choices(consts, argc)
        targets.append(((preds[i], args), 1))
        # Fail on out of context predicate with same arguments
        targets.append(((preds[-1], args), 0))
    else:
      # Some ground instances
      k = 2 if R.random() < 0.5 else 1
      pred = (preds[i], choices(consts[:-1], k))
      ctx.append([pred])
  output(ctx, targets)

def nstep_deduction(ctx_size, steps, negation=False):
  assert steps >= 1
  assert ctx_size >= (steps + 2)
  preds = r_preds(ctx_size*2+steps)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  i, pidx = 0, 0
  prefix = '-' if negation else ''
  while i < ctx_size:
    rtype = R.randrange(2 if i == 0 else 3)
    if rtype == 0:
      # Double variable swap deduction rules
      vs = R.sample(var, 2)
      ctx.append([(preds[pidx], vs), (prefix+preds[pidx+1], vs[::-1])])
      if i == 0:
        # Add the n steps
        swapc = 1
        for j in range(steps-1):
          vs = R.sample(var, 2)
          toswap = R.random() < 0.5 # Do we swap again?
          args = vs[::-1] if toswap else vs
          ctx.append([(preds[pidx+j+1], vs), (preds[pidx+j+2], args)])
          swapc += int(toswap)
        # Add the ground case
        args = R.sample(consts[:-1], 2)
        ctx.append([(preds[pidx+steps], args)])
        i += steps
        args = args if swapc % 2 == 0 else args[::-1]
        targets.append(((preds[pidx], args), 1-int(negation)))
        targets.append(((preds[pidx], args[::-1]), int(negation)))
        pidx += steps-1
      pidx += 2
    elif rtype == 1:
      # Double variable non-swap deduction rules
      # Single variable deduction rules
      argc = R.randint(1, 2)
      vs = R.sample(var, argc)
      ctx.append([(preds[pidx], vs), (prefix+preds[pidx+1], vs)])
      if i == 0:
        # Add the n steps
        for j in range(steps-1):
          vs = R.sample(var, argc)
          ctx.append([(preds[pidx+j+1], vs), (preds[pidx+j+2], vs)])
        args = choices(consts[:-1], argc)
        # Add the ground case
        ctx.append([(preds[pidx+steps], args)])
        i += steps
        targets.append(((preds[pidx], args), 1-int(negation)))
        # Fail on non-matching constant
        args = args.copy()
        args[R.randrange(len(args))] = consts[-1]
        ctx.append([(preds[-1], args)]) # Decoy rule
        i += 1
        targets.append(((preds[pidx], args), int(negation)))
        pidx += steps-1
      pidx += 2
    else:
      # Ground instances
      ctx.append([(preds[pidx], choices(consts, 1))])
      pidx += 1
    prefix = ''
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

def logical_and(ctx_size, negation=False):
  """Logical AND with optional negation: p(X):-q(X);r(X)."""
  assert ctx_size >= (4 if negation else 3)
  preds = r_preds(ctx_size*3+1)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  i, pidx = 0, 0
  while i < ctx_size:
    rtype = R.randrange(1 if i == 0 else 2)
    if rtype == 0:
      argc = R.randint(1, 2)
      # Double variable AND with different vars
      # Single variable AND
      vs = R.sample(var, argc)
      ctx.append([(preds[pidx], vs),
                  (preds[pidx+1], vs[:1]),
                  (preds[pidx+2], vs[1:] or vs)])
      if i == 0:
        ridx = R.randint(1, 2)
        if negation:
          # Add negation to random predicate in body
          pred = ctx[-1][ridx]
          ctx[-1][ridx] = ('-' + pred[0], pred[1])
        # Add the ground cases
        args = choices(consts[:-1], argc)
        ctx.append([(preds[pidx+1], args[:1])])
        ctx.append([(preds[pidx+2], args[1:] or args)])
        i += 2
        if negation and argc == 1:
          # Add the non-matching case to prove the rule
          ctx.append([(preds[pidx+(3-ridx)], [consts[-1]])])
          i += 1
        # Successful case
        targets.append(((preds[pidx], args), 1-int(negation)))
        # Fail on non-matching constant
        args = args.copy()
        args[min(ridx-1, len(args)-1)] = consts[-1]
        targets.append(((preds[pidx], args), int(negation)))
      pidx += 3
    else:
      # Some other ground cases
      k = 2 if R.random() < 0.5 else 1
      ctx.append([(preds[pidx], choices(consts, k))])
      pidx += 1
    i += 1
  output(ctx, targets)

def gen_task6(ctx_size):
  """Logical AND: p(X):-q(X);r(X)."""
  logical_and(ctx_size)

def logical_or(ctx_size, negation=False):
  """Logical OR with optional negation: p(X):-q(X).p(X):-r(X)."""
  assert ctx_size >= 4
  preds = r_preds(ctx_size*3+1)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  i, pidx = 0, 0
  while i < ctx_size:
    rtype = R.randrange(1 if i == 0 else 2)
    if rtype == 0:
      # Double or single variable OR
      argc = R.randint(1, 2)
      vs = R.sample(var, argc)
      swap = R.random() < 0.5
      prefix = '-' if negation and i == 0 else ''
      ctx.append([(preds[pidx], vs), (prefix + preds[pidx+1], vs if swap else vs[::-1])])
      if i == 0:
        # Add the extra branching rule
        ctx.append([(preds[pidx], vs), (preds[pidx+2], vs)])
        # Add ground cases
        args = choices(consts[:-1], argc)
        ctx.append([(preds[pidx+1], args if swap else args[::-1])])
        argso = choices(consts[:-1], argc)
        ctx.append([(preds[pidx], argso)])
        i += 3
        if R.random() < 0.2 and not negation:
          # Shortcut case
          targets.append(((preds[pidx], argso), 1))
        else:
          # Follow only one of the rules
          targets.append(((preds[pidx], args), 1-int(negation)))
        # Fail on non-matching constant
        args = args.copy()
        args[R.randrange(len(args))] = consts[-1]
        targets.append(((preds[pidx], args), int(negation)))
      pidx += 3
    else:
      # Some other ground cases
      k = 2 if R.random() < 0.5 else 1
      ctx.append([(preds[pidx], choices(consts, k))])
      pidx += 1
    i += 1
  output(ctx, targets)

def gen_task7(ctx_size):
  """Logical OR: p(X):-q(X).p(X):-r(X)."""
  logical_or(ctx_size)

def gen_task8(ctx_size):
  """Transitive case: p(X,Y):-q(X,Z);r(Z,Y)."""
  assert ctx_size >= 5
  preds = r_preds(ctx_size*3+1)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  i, pidx = 0, 0
  while i < ctx_size:
    rtype = R.randrange(1 if i == 0 else 2)
    if rtype == 0:
      # Existential variable with single choice
      vs = R.sample(var, 3)
      ctx.append([(preds[pidx], [vs[0], vs[2]]),
                  (preds[pidx+1], vs[:2]),
                  (preds[pidx+2], vs[1:])])
      if i == 0:
        # Add matching ground cases
        args = choices(consts[:-1], 3)
        ctx.append([(preds[pidx+1], args[:2])])
        ctx.append([(preds[pidx+2], args[1:])])
        # Add non-matching ground cases
        argso = choices(consts[:-1], 3)
        argso.insert(R.randint(1, 2), consts[-1])
        ctx.append([(preds[pidx+1], argso[:2])])
        ctx.append([(preds[pidx+2], argso[2:])])
        i += 4
        # Successful case
        targets.append(((preds[pidx], [args[0], args[2]]), 1))
        # Fail on half-matching existential
        targets.append(((preds[pidx], [argso[0], argso[3]]), 0))
      pidx += 3
    else:
      # Some other ground cases
      k = 2 if R.random() < 0.5 else 1
      ctx.append([(preds[pidx], choices(consts, k))])
      pidx += 1
    i += 1
  output(ctx, targets)

def gen_task9(ctx_size):
  """Single step deduction with NBF: p(X):--q(X)."""
  nstep_deduction(ctx_size, 1, True)

def gen_task10(ctx_size):
  """Double step deduction with NBF: p(X):--q(X).q(X):-r(X)."""
  nstep_deduction(ctx_size, 2, True)

def gen_task11(ctx_size):
  """Logical AND with NBF: p(X):-q(X);-r(X)."""
  logical_and(ctx_size, True)

def gen_task12(ctx_size):
  """Logical OR with NBF: p(X):--q(X).p(X):-r(X)."""
  logical_or(ctx_size, True)

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
