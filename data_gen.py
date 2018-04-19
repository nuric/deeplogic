"""Data generation script for logic programs."""
import argparse
import random as R

# Symbol Pool
CONST_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
VAR_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRED_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
EXTRA_SYMBOLS = "-,()"

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

def r_symbols(size, symbols, length, used=None):
  """Return unique random from given symbols."""
  if length == 1 and not used:
    return R.sample(symbols, size)
  rset, used = set(), set(used or [])
  while len(rset) < size:
    s = r_string(symbols, length)
    if s not in used:
      rset.add(s)
  return list(rset)

def r_consts(size, used=None):
  """Return size many unique constants."""
  return r_symbols(size, CONST_SYMBOLS, ARGS.constant_length, used)

def r_vars(size, used=None):
  """Return size many unique variables."""
  return r_symbols(size, VAR_SYMBOLS, ARGS.variable_length, used)

def r_preds(size, used=None):
  """Return size many unique predicates."""
  return r_symbols(size, PRED_SYMBOLS, ARGS.predicate_length, used)

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

def gen_task(context, targets, upreds):
  """Fill context with random preds and output program."""
  n_tofill = ARGS.context_size - len(context)
  assert n_tofill >= 0, "Task context requires larger size."
  # Fill with random rules up to certain task
  ctx = context.copy() # Don't modify the original context
  for _ in range(n_tofill):
    task = "gen_task" + str(R.randint(1, ARGS.task))
    ctx.append(globals()[task](upreds))
  output(ctx, targets)

def fail_pred(context, pred, upreds, uconsts, psuccess=0.0):
  """Fail a ground case predicate given context."""
  # Maybe succeed by adding to context
  if R.random() < psuccess:
    context.append([pred])
  elif R.random() < 0.7:
    # The constant doesn't match
    args = pred[1].copy()
    args[R.randrange(len(args))] = r_consts(1, uconsts)[0]
    context.append([(pred[0], args)])
    # The predicate doesn't match
    context.append([(r_preds(1, upreds)[0], pred[1])])
  # The predicate doesn't appear at all

def gen_task1(upreds=None):
  """Ground instances only: p(a).q(c,b)."""
  # One or two argument predicate
  preds = r_preds(2, upreds)
  args = r_consts(R.randint(1, 2))
  rule = [(preds[0], args)]
  if upreds:
    return rule
  ctx = [rule]
  # Successful case when query appears in context
  targets = [(ctx[0][0], 1)]
  # Fail case
  args = r_consts(R.randint(1, 2))
  fpred = (preds[1], args)
  fail_pred(ctx, fpred, preds, args)
  targets.append((fpred, 0))
  gen_task(ctx, targets, preds)

def gen_task2(upreds=None):
  """Variablised facts only: p(X).q(X,Y)."""
  preds = r_preds(2, upreds)
  ctx, targets = list(), list()
  if R.random() < 0.5:
    # Double variable same argument
    v = r_vars(1)[0]
    rule = [(preds[0], [v, v])]
    if upreds:
      return rule
    ctx.append(rule)
    # Successful double variable grounding
    cs = r_consts(2)
    c = R.choice(cs)
    targets.append(((preds[0], [c, c]), 1))
    # Fail on non-unique variable grounding
    targets.append(((preds[0], cs), 0))
  else:
    # Double variable different argument
    # Single variable argument
    argc = R.randint(1, 2)
    args = r_vars(argc)
    rule = [(preds[0], args)]
    if upreds:
      return rule
    ctx.append(rule)
    # Successful unique argument grounding
    args = choices(r_consts(2), argc)
    targets.append(((preds[0], args), 1))
    # Fail on out of context predicate with same arguments
    targets.append(((preds[1], args), 0))
  gen_task(ctx, targets, preds)

def nstep_deduction(steps, negation=False, upreds=None):
  assert steps >= 1, "Need at least 1 step deduction."
  preds = r_preds(2 if upreds else 3+steps, upreds)
  consts = r_consts(2)
  ctx, targets = list(), list()
  prefix = '-' if negation else ''
  if R.random() < 0.5:
    # Double variable swap deduction rules
    vs = r_vars(2)
    rule = [(preds[0], vs), (prefix+preds[1], vs[::-1])]
    if upreds:
      return rule
    ctx.append(rule)
    # Add the n steps
    swapc = 1
    for j in range(steps-1):
      vs = r_vars(2)
      toswap = R.random() < 0.5 # Do we swap again?
      args = vs[::-1] if toswap else vs
      ctx.append([(preds[j+1], vs), (preds[j+2], args)])
      swapc += int(toswap)
    # Add the ground case
    args = r_consts(2)
    ctx.append([(preds[steps], args)])
    args = args if swapc % 2 == 0 else args[::-1]
    targets.append(((preds[0], args), 1-int(negation)))
    targets.append(((preds[0], args[::-1]), int(negation)))
    gen_task(ctx, targets, preds)
  else:
    # Double variable non-swap deduction rules
    # Single variable deduction rules
    argc = R.randint(1, 2)
    vs = r_vars(argc)
    rule = [(preds[0], vs), (prefix+preds[1], vs)]
    if upreds:
      return rule
    ctx.append(rule)
    # Add the n steps
    for j in range(steps-1):
      vs = r_vars(argc)
      ctx.append([(preds[j+1], vs), (preds[j+2], vs)])
    args = choices(r_consts(2), argc)
    # Add the ground case
    cctx = ctx.copy()
    spred = (preds[steps], args)
    cctx.append([spred])
    cctx.append([(preds[-1], args)])
    targets = [((preds[0], args), 1-int(negation))]
    gen_task(cctx, targets, preds)
    # Add failure case
    fail_pred(ctx, spred, preds, args)
    targets = [((preds[0], args), int(negation))]
    gen_task(ctx, targets, preds)

def gen_task3(upreds=None):
  """Single step deduction: p(X):-q(X)."""
  return nstep_deduction(1, upreds=upreds)

def gen_task4(upreds=None):
  """Double step deduction: p(X):-q(X).q(X):-r(X)."""
  return nstep_deduction(2, upreds=upreds)

def gen_task5(upreds=None):
  """Triple step deduction."""
  return nstep_deduction(3, upreds=upreds)

def logical_and(negation=False, upreds=None):
  """Logical AND with optional negation: p(X):-q(X);r(X)."""
  preds = r_preds(3, upreds)
  argc = R.randint(1, 2)
  # Double variable AND with different vars
  # Single variable AND
  vs = r_vars(argc)
  rule = [(preds[0], vs),
          (preds[1], vs[:1]),
          (preds[2], vs[1:] or vs)]
  if upreds:
    return rule
  ctx = [rule]
  # Create the ground arguments
  args = choices(r_consts(2), argc)
  prem1 = (preds[1], args[:1])
  prem2 = (preds[2], args[1:] or args)
  prems = [prem1, prem2]
  if negation:
    # Add negation to random predicate in body
    ridx = R.randrange(2)
    p, pargs = ctx[-1][ridx+1]
    ctx[-1][ridx+1] = ('-' + p, pargs)
    # Successful case when negation fails
    cctx = ctx.copy()
    fail_pred(cctx, prems[ridx], preds, args)
    cctx.append([prems[1-ridx]])
    targets = [((preds[0], args), 1)]
    gen_task(cctx, targets, preds)
    # Fail one premise randomly
    fidx = R.randrange(2)
    if ridx == fidx:
      # To fail negation add ground instance
      ctx.append([prems[ridx]])
      # Succeed other with some probability
      fail_pred(ctx, prems[1-ridx], preds, args, 0.8)
    else:
      # Fail non-negated premise
      fail_pred(ctx, prems[1-ridx], preds, args)
      # Still succeed negation
      fail_pred(ctx, prems[ridx], preds, args)
    targets = [((preds[0], args), 0)]
    gen_task(ctx, targets, preds)
  else:
    # Create successful context
    cctx = ctx.copy()
    cctx.extend([[p] for p in prems])
    targets = [((preds[0], args), 1)]
    gen_task(cctx, targets, preds)
    # Fail one premise randomly
    fidx = R.randrange(2)
    fail_pred(ctx, prems[fidx], preds, args)
    # Succeed the other with some probability
    fail_pred(ctx, prems[1-fidx], preds, args, 0.8)
    targets = [((preds[0], args), 0)]
    gen_task(ctx, targets, preds)

def gen_task6(upreds=None):
  """Logical AND: p(X):-q(X);r(X)."""
  return logical_and(upreds=upreds)

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
        if negation:
          args = R.sample(consts[:-1], argc*2)
        else:
          args = choices(consts[:-1], argc*2)
        argso = args[len(args)//2:]
        args = args[:len(args)//2]
        ctx.append([(preds[pidx+1], args if swap else args[::-1])])
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

def gen_task9(upreds=None):
  """Single step deduction with NBF: p(X):--q(X)."""
  return nstep_deduction(1, True, upreds)

def gen_task10(upreds=None):
  """Double step deduction with NBF: p(X):--q(X).q(X):-r(X)."""
  return nstep_deduction(2, True, upreds)

def gen_task11(upreds=None):
  """Logical AND with NBF: p(X):-q(X);-r(X)."""
  return logical_and(True, upreds)

def gen_task12(ctx_size):
  """Logical OR with NBF: p(X):--q(X).p(X):-r(X)."""
  logical_or(ctx_size, True)

def gen_task0(ctx_size):
  """Generate an ILP task example."""
  assert ctx_size >= 1
  argc = 1
  goal= 'f'
  premise = 'b'
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+2)
  var = r_vars(ctx_size)
  ctx, targets = list(), list()
  # Generate according to goal <- premise
  i, pidx = 0, 0
  while i < ctx_size:
    if i == 0:
      args = choices(consts[:-1], argc)
      # Add the successful ground case
      ctx.append([(premise, args)])
      targets.append(((goal, args), 1))
      # Fail on non-matching constant
      args = args.copy()
      args[R.randrange(len(args))] = consts[-1]
      ctx.append([(preds[pidx], args)])
      pidx += 1
      targets.append(((goal, args), 0))
      # Add padding length dummy rule
      vs = choices(var, argc)
      ctx.append([(preds[pidx], vs), (preds[pidx+1], vs)])
      i += 2
    else:
      # Fill with noise, ground atoms
      ctx.append([(preds[pidx], choices(consts, argc))])
    i += 1
    pidx += 1
  output(ctx, targets)

if __name__ == '__main__':
  # pylint: disable=line-too-long
  # Arguments
  parser = argparse.ArgumentParser(description="Generate logic program data.")
  parser.add_argument("-t", "--task", default=1, type=int, help="The task to generate.")
  parser.add_argument("-s", "--size", default=1, type=int, help="Number of programs to generate.")
  # Configuration parameters
  parser.add_argument("-cs", "--context_size", default=6, type=int, help="Size of program context.")
  parser.add_argument("-cl", "--constant_length", default=1, type=int, help="Length of constants.")
  parser.add_argument("-vl", "--variable_length", default=1, type=int, help="Length of variables.")
  parser.add_argument("-pl", "--predicate_length", default=1, type=int, help="Length of predicates.")
  parser.add_argument("-sf", "--shuffle_context", action="store_true", help="Shuffle context before output.")
  # Task specific options
  parser.add_argument("-ns", "--nstep", type=int, help="Generate nstep deduction programs.")
  ARGS = parser.parse_args()

  # Generate given task
  task = "gen_task" + str(ARGS.task)
  for _ in range(ARGS.size):
    if ARGS.nstep:
      nstep_deduction(ARGS.context_size, ARGS.nstep)
    else:
      globals()[task]()
