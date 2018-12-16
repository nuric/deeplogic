"""Tree based data generation script for logic programs."""
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
NEG_PREFIX = '-'
TARGET_T = "? {} {}"

# pylint: disable=line-too-long,too-many-arguments,too-many-statements

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
    s = r_string(symbols, R.randint(1, length))
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
  return PRED_T.format(pred[0], ARG_SEP.join(pred[1:]))

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
  # context: [[('p', 'a', 'b')], ...]
  # targets: [(('p', 'a', 'b'), 1), ...]
  if ARGS.shuffle_context:
    R.shuffle(context)
  print('\n'.join([write_r(c) for c in context]))
  for t, v in targets:
    print(TARGET_T.format(write_r([t]), v))

def cv_mismatch(consts):
  """Returns a possible mismatching variable binding for given constants."""
  if len(consts) <= 1 or len(set(consts)) == 1:
    return list()
  # We know some constant is different
  # [a,b,a,c] -> [X,Y,Y,Z]
  # [a,b] -> [X,X] are mismatches
  # assign same variables to different constants
  vs = r_vars(len(consts)-1) # [X,Y,Z,..]
  for i, c in enumerate(consts[1:]):
    if c != consts[0]:
      # we haven't seen it before
      vs.insert(i+1,vs[0])
      break
  assert len(vs) == len(consts)
  return vs

def cv_match(consts):
  """Returns a possible matching variable binding for given constants."""
  if len(consts) <= 1:
    return r_vars(len(consts))
  # We want to *randomly* assing the same variable to same constants
  # [a,a,b] -> [X,Y,Z] -> [X,X,Y]
  vs = r_vars(len(consts))
  cvmap = dict()
  for i, c in enumerate(consts):
    if c in cvmap:
      if R.random() < 0.5:
        vs[i] = cvmap[c] # assign the same variable
      # otherwise get a unique variable
    else:
      cvmap[c] = vs[i]
  assert len(vs) == len(consts)
  return vs

def generate(depth=0, context=None, target=None, success=None,
             upreds=None, uconsts=None, stats=None):
  """Generate tree based logic program."""
  ctx = context or list()
  args = target[1:] if target else [r_consts(1)[0] for _ in range(ARGS.arity)]
  t = target or [r_preds(1)[0]] + [R.choice(args) for _ in range(R.randint(1, ARGS.arity))]
  arity = len(t[1:])
  succ = success if success is not None else R.choice((True, False))
  upreds = upreds or set([t[0]])
  uconsts = uconsts or set(t[1:])
  stats = stats or dict()

  # Create rule OR branching
  num_rules = R.randint(1, ARGS.max_or_branch)
  stats.setdefault('or_num', list()).append(num_rules)
  # If the rule succeeds than at least one branch must succeed
  succs = [R.choice((True, False)) for _ in range(num_rules)] \
          if succ else [False]*num_rules # otherwise all branches must fail
  if succ and not any(succs):
    # Ensure at least one OR branch succeeds
    succs[R.randrange(len(succs))] = True
  # Rule depths randomised between 0 to max depth
  depths = [R.randint(0, depth) for _ in range(num_rules)]
  if max(depths) != depth:
    depths[R.randrange(num_rules)] = depth
  # print("HERE:", num_rules, succs, depths, t)

  # Generate OR branches
  is_tadded = False
  for child_depth, child_succ in zip(depths, succs):
    # Base case
    if child_depth == 0:
      if R.random() < 0.20:
        # The constant doesn't match
        args = t[1:]
        args[R.randrange(len(args))] = r_consts(1, uconsts)[0]
        uconsts.update(args)
        ctx.append([[t[0]] + args])
      if R.random() < 0.20:
        # The predicate doesn't match
        p = r_preds(1, upreds)[0]
        upreds.add(p)
        ctx.append([[p,] + t[1:]])
      if R.random() < 0.20:
        # The arity doesn't match
        ctx.append([[t[0]] + t[1:] + [R.choice(t[1:] + r_consts(arity))]])
      if R.random() < 0.20:
        # The variables don't match
        vs = cv_mismatch(t[1:])
        if vs:
          ctx.append([[t[0]] + vs])
      # The predicate doesn't appear at all
      if child_succ:
        if R.random() < 0.5:
          # p(X). case
          ctx.append([[t[0]] + cv_match(t[1:])])
        elif not is_tadded:
          # ground case
          ctx.append([t])
          is_tadded = True
      continue
    # Recursive case
    num_body = R.randint(1, ARGS.max_and_branch)
    stats.setdefault('body_num', list()).append(num_body)
    negation = [R.choice((True, False)) for _ in range(num_body)] \
               if ARGS.negation else [False]*num_body
    # Compute recursive success targets
    succ_targets = [R.choice((True, False)) for _ in range(num_body)] \
                   if not child_succ else [not n for n in negation]
    if not child_succ:
      # Ensure a failed target
      ri = R.randrange(len(succ_targets))
      # succeeding negation fails this, vice versa
      succ_targets[ri] = negation[ri]
    # Create rule
    body_preds = r_preds(num_body, upreds)
    upreds.update(body_preds)
    lit_vars = cv_match(t[1:])
    if not child_succ and R.random() < 0.5:
      # Fail due to variable pattern mismatch
      vs = cv_mismatch(t[1:])
      if vs:
        lit_vars = vs
        succ_targets = [R.choice((True, False)) for _ in range(num_body)]
    lit_vars.extend([r_vars(1)[0] for _ in range(ARGS.unbound_vars)])
    rule = [[t[0]]+lit_vars[:arity]]
    vcmap = {lit_vars[i]:t[i+1] for i in range(arity)}
    # Compute child targets
    child_targets = list()
    for i in range(num_body):
      R.shuffle(lit_vars)
      child_arity = R.randint(1, arity)
      pred = [body_preds[i]] + lit_vars[:child_arity]
      rule.append([(NEG_PREFIX if negation[i] else "") + pred[0]] + pred[1:])
      vs = [vcmap.get(v, r_consts(1, uconsts)[0]) for v in lit_vars[:child_arity]]
      child_targets.append([pred[0]]+vs)
    ctx.append(rule)
    # Recurse
    for child_t, s in zip(child_targets, succ_targets):
      generate(child_depth-1, ctx, child_t, s, upreds, uconsts, stats)
  return ctx, [(t, int(succ))], stats

if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(description="Generate logic program data.")
  parser.add_argument("-d", "--depth", default=0, type=int, help="The depth of the logic program.")
  parser.add_argument("-mob", "--max_or_branch", default=1, type=int, help="Upper bound on number of branches.")
  parser.add_argument("-mab", "--max_and_branch", default=1, type=int, help="Upper bound on number of branches.")
  parser.add_argument("-s", "--size", default=1, type=int, help="Number of programs to generate.")
  # Configuration parameters
  parser.add_argument("-uv", "--unbound_vars", default=0, type=int, help="Number of unbound variables.")
  parser.add_argument("-ar", "--arity", default=2, type=int, help="Upper bound on arity of literals.")
  parser.add_argument("-n", "--negation", action="store_true", help="Use negation by failure.")
  parser.add_argument("-cl", "--constant_length", default=2, type=int, help="Length of constants.")
  parser.add_argument("-vl", "--variable_length", default=1, type=int, help="Length of variables.")
  parser.add_argument("-pl", "--predicate_length", default=2, type=int, help="Length of predicates.")
  parser.add_argument("-sf", "--shuffle_context", action="store_true", help="Shuffle context before output.")
  ARGS = parser.parse_args()

  for _ in range(ARGS.size):
    context_out, targets_out, _ = generate(depth=ARGS.depth)
    output(context_out, targets_out)
