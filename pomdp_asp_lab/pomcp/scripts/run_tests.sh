RUNS=30  # number of runs per experiment
POMCP="../build/pomcp" #executable
DIR="./rocksample_results" #to save results
mkdir -p $DIR

### ROCKSAMPLE
#PARAMETERS:
#--size : grid size
SIZE=12;
#--number : num rocks
ROCKS=4;
# -- mindouble / maxdouble: number of particles and simulations (2^N)
N=13;

# WITH RULES
TK=3; #UCT SEARCH - 1: only legal; 2: preferred; 3: rules (OURS);
RK=4; #ROLLOUT - 1: random; 2: preferred; 4: rules (OURS)
$POMCP --problem rocksample --size $((SIZE)) --number $((ROCKS)) --soft=true --val 100 --timeout 100000000 --mindouble $((N)) --maxdouble $((N)) --runs $((RUNS)) --xes 1 --rolloutknowledge=$((RK)) --treeknowledge=$((TK)) --smarttreecount 10 --smarttreevalue 20;
mv ./log.xes "$DIR/rocksample_$((N))part_$((SIZE))size_$((ROCKS))rocks_r$((RK))_t$((TK)).xes";

# WITHOUT RULES
TK=1; #UCT SEARCH - 1: only legal; 2: preferred; 3: rules (OURS);
RK=1; #ROLLOUT - 1: random; 2: preferred; 4: rules (OURS)
$POMCP --problem rocksample --size $((SIZE)) --number $((ROCKS)) --soft=true --val 100 --timeout 100000000 --mindouble $((N)) --maxdouble $((N)) --runs $((RUNS)) --xes 1 --rolloutknowledge=$((RK)) --treeknowledge=$((TK)) --smarttreecount 10 --smarttreevalue 20;
mv ./log.xes "$DIR/rocksample_$((N))part_$((SIZE))size_$((ROCKS))rocks_r$((RK))_t$((TK)).xes";
