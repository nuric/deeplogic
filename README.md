# DeepLogic
This repository contains the code for the paper [DeepLogic: Towards End-to-End Differentiable Logical Reasoning](https://arxiv.org/abs/1805.07433). The goal is to train a fixed architecture neural network to perform logical reasoning in the domain of logic programs. There are numerous attempts at *combining* machine learning with symbolic reasoning to get the best of both world; however, in this work we are interested *to what extent does machine learning already encompass symbolic reasoning* by learning it from scratch. The repository also incorporates extra code for research as part of future work.

## Generating Data
All the data in the current work is generated using a single pure Python script `data_gen.py` with the following options:

```bash
# Example usage
python3 data_gen.py -h

	usage: data_gen.py [-h] [-t TASK] [-s SIZE] [-ns NOISE_SIZE]
										 [-cl CONSTANT_LENGTH] [-vl VARIABLE_LENGTH]
										 [-pl PREDICATE_LENGTH] [-sf] [--nstep NSTEP]

	Generate logic program data.

	optional arguments:
		-h, --help            show this help message and exit
		-t TASK, --task TASK  The task to generate.
		-s SIZE, --size SIZE  Number of programs to generate.
		-ns NOISE_SIZE, --noise_size NOISE_SIZE
													Size of added noise rules.
		-cl CONSTANT_LENGTH, --constant_length CONSTANT_LENGTH
													Length of constants.
		-vl VARIABLE_LENGTH, --variable_length VARIABLE_LENGTH
													Length of variables.
		-pl PREDICATE_LENGTH, --predicate_length PREDICATE_LENGTH
													Length of predicates.
		-sf, --shuffle_context
													Shuffle context before output.
		--nstep NSTEP         Generate nstep deduction programs.

# Example program
python3 data_gen.py -t 4 -s 1 -ns 2 -cl 4 -pl 4

	i(R):-ewh(R).
	ewh(S):-yxp(S).
	yxp(qf).
	yxp(l).
	run(jiy).
	y(J,M):-v(M,J).
	? i(qf). 1
	i(R):-ewh(R).
	ewh(S):-yxp(S).
	t(zka,h).
	xbq(V,M):-dh(M,V).
	? i(qf). 0
```
This script generates a pair of positive and negative answers over the same query. The details of the tasks and how they fail are explained in the paper and also in the code. There is also a utility script `gen_task.sh` that wraps around the `data_gen.py` script to generate and save the programs into files under the data folder:

```bash
mkdir data
./gen_task.sh all # will generate single file with all tasks
./gen_task.sh acc # will generate accumulating files up to task 5, so 1, 1 2, 1 2 3 etc
./gen_task.sh iter # will generate based on number of iterations, so 1 2, 1 2 3 7 9 12, etc
./gen_task.sh eval_single # will generate evaluate / test sets with increasing symbol and noise lenghts
```

### Generic Logic Programs
As part of future work, this repository contains a more advanced, complicated logic program generation script `gen_logic.py` that recursively generates mixed logic programs up to a certain depth. It combines all the elements of the tasks into a single script and generates programs at random. It is also a pure Python script with no dependencies:

```bash
# Usage
python3 gen_logic.py -h
	usage: gen_logic.py [-h] [-d DEPTH] [-mob MAX_OR_BRANCH] [-mab MAX_AND_BRANCH]
											[-s SIZE] [-uv UNBOUND_VARS] [-ar ARITY] [-n]
											[-cl CONSTANT_LENGTH] [-vl VARIABLE_LENGTH]
											[-pl PREDICATE_LENGTH] [-sf]

	Generate logic program data.

	optional arguments:
		-h, --help            show this help message and exit
		-d DEPTH, --depth DEPTH
													The depth of the logic program.
		-mob MAX_OR_BRANCH, --max_or_branch MAX_OR_BRANCH
													Upper bound on number of branches.
		-mab MAX_AND_BRANCH, --max_and_branch MAX_AND_BRANCH
													Upper bound on number of branches.
		-s SIZE, --size SIZE  Number of programs to generate.
		-uv UNBOUND_VARS, --unbound_vars UNBOUND_VARS
													Number of unbound variables.
		-ar ARITY, --arity ARITY
													Upper bound on arity of literals.
		-n, --negation        Use negation by failure.
		-cl CONSTANT_LENGTH, --constant_length CONSTANT_LENGTH
													Length of constants.
		-vl VARIABLE_LENGTH, --variable_length VARIABLE_LENGTH
													Length of variables.
		-pl PREDICATE_LENGTH, --predicate_length PREDICATE_LENGTH
													Length of predicates.
		-sf, --shuffle_context
													Shuffle context before output.
# Example program
python3 gen_logic.py -d 1 -mob 3 -mab 3 -s 1 -ar 3 -n -cl 3 -pl 3

	uf(N,N,N):-m(N,N,N);-ejt(N);ywe(N).
	m(a,a,a,irs).
	ejt(ck).
	ek(a).
	ywe(kge).
	uf(O,O,U):--nwq(U,O,O);vh(U).
	po(a,a,a).
	nwq(a,a,a).
	vh(pfv).
	? uf(a,a,a). 0
```
Unlike the previous script, `gen_logic.py` generates at either a positive or negative result at random.

## Training
There is a training script `train.py` that encapsulates training models defined in `models` directory. The models are built using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) which can be installed using:

```bash
pip3 install --no-cache-dir --upgrade -r requirements.py
```

Then after the data is generated, any model can be trained using:
```bash
# Usage
python3 train.py -h

	usage: train.py [-h] [-md MODEL_DIR] [--dim DIM] [-d]
									[-ts [TASKS [TASKS ...]]] [-e EPOCHS] [-s] [-i]
									[-its ITERATIONS] [-bs BATCH_SIZE] [-p]
									model model_file

	Train logic-memnn models.

	positional arguments:
		model                 The name of the module to train.
		model_file            Model filename.

	optional arguments:
		-h, --help            show this help message and exit
		-md MODEL_DIR, --model_dir MODEL_DIR
													Model weights directory ending with /.
		--dim DIM             Latent dimension.
		-d, --debug           Only predict single data point.
		-ts [TASKS [TASKS ...]], --tasks [TASKS [TASKS ...]]
													Tasks to train on, blank for all tasks.
		-e EPOCHS, --epochs EPOCHS
													Number of epochs to train.
		-s, --summary         Dump model summary on creation.
		-i, --ilp             Run ILP task.
		-its ITERATIONS, --iterations ITERATIONS
													Number of model iterations.
		-bs BATCH_SIZE, --batch_size BATCH_SIZE
													Training batch_size.
		-p, --pad             Pad context with blank rule.

# Example training
mkdir weights
python3 train.py imasm curr_imasm64 -p -ts 0 1 2 -its 2 # train imasm on tasks 0 1 2 with 2 iterations of the network
```

There is also an interactive debug mode and the corresponding attention maps along with the output is displayed. This feature is useful for understanding the iterative steps and interpreting the attention maps.
```bash
python3 train.py imasm curr_imasm64 -p -ts 0 1 2 -its 3 -d # same command as before but add -d for debug, can also change iterations

	CTX: p(X):-q(X).q(Y):-t(Y).t(a).q(b).
	Q: p(a).
	[[0.99750346 0.00000391 0.00000503 0.00000388 0.00241059 0.00007312]] # iteration one attention map
	[[0.00214907 0.9691639  0.00007983 0.00981691 0.01855759 0.0002328 ]] # last 2 columns are blank and null sentinel
	[[0.0021707  0.00218101 0.97639084 0.00023426 0.01878748 0.00023554]]
	[[0.9999949]]
	OUT: 0.9999948740005493
```

## Evaluating
Similar to training, there is a corresponding `eval.py` script that runs the models on the evaluation / test data as well as plots charts such as attention maps. It can be used as follows:
```bash
# Usage
python3 eval.py
	usage: eval.py [-h] [-md MODEL_DIR] [--dim DIM] [-f FUNCTION] [--outf OUTF]
								 [-s] [-its ITERATIONS] [-bs BATCH_SIZE] [-p]
								 model model_file

	Evaluate logic-memnn models.

	positional arguments:
		model                 The name of the module to train.
		model_file            Model filename.

	optional arguments:
		-h, --help            show this help message and exit
		-md MODEL_DIR, --model_dir MODEL_DIR
													Model weights directory ending with /.
		--dim DIM             Latent dimension.
		-f FUNCTION, --function FUNCTION
													Function to run.
		--outf OUTF           Plot to output file instead of rendering.
		-s, --summary         Dump model summary on creation.
		-its ITERATIONS, --iterations ITERATIONS
													Number of model iterations.
		-bs BATCH_SIZE, --batch_size BATCH_SIZE
													Evaluation batch_size.
		-p, --pad             Pad context with blank rule.

# If the evaluation data is generated
python3 eval.py imasm curr_imasm64 # will evaluate on data/test_{set}_task{num}.txt ex. test_easy_task1.txt

# Example to plot attention map
python3 eval.py imasm curr_imasm64 -f plot_attention
```

## FAQ

 - **Why is there code such as ILP that is not included in the paper?** The repository contains extra code towards future work. They are mainly just quick skeletons, experiments of extra ideas and attempts. They are included in the repository to keep everything in one place and consistent across any future work.
 - **Why is the source code 2 space indented?** The answer is a combination of personal style and to stop direct copy-paste from other resources. The code is linted using [PyLint](https://www.pylint.org/) although there are cases it is disabled on purpose.

## Built With

 -[Keras](https://keras.io/) - library used to implement models
 -[TensorFlow](https://www.tensorflow.org/) - library used to train and run models
 -[Matplotlib](https://matplotlib.org/) - main plotting library
 -[seaborn](https://seaborn.pydata.org/) - helper plotting library for some charts
 -[NumPy](http://www.numpy.org/) - main numerical library for data vectorisation
 -[Pandas](https://pandas.pydata.org/) - helper data manipulation library
