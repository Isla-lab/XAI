#include "experiment.h"
#include "mcts.h"
#include "network.h"
#include "rocksample.h"

#include <boost/program_options.hpp>
#include <memory>

using namespace std;
using namespace boost::program_options;

void UnitTests() {
    cout << "Testing UTILS" << endl;
    UTILS::UnitTest();
    cout << "Testing COORD" << endl;
    COORD::UnitTest();
    cout << "Testing MCTS" << endl;
    MCTS::UnitTest();
}

void disableBufferedIO() {
    setbuf(stdout, nullptr);
    setbuf(stdin, nullptr);
    setbuf(stderr, nullptr);
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stdin, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
}

int main(int argc, char *argv[]) {
    MCTS::PARAMS searchParams;
    EXPERIMENT::PARAMS expParams;
    SIMULATOR::KNOWLEDGE knowledge;
    string problem, outputfile, policy;
    int size, number, ghosts, treeknowledge = 0, rolloutknowledge = 1,
                      smarttreecount = 10;
    double smarttreevalue = 1.0;
    int random_seed = 12345678;
    std::string shield_file;

    double W = 0.0;
    bool complex_shield = false, xes_log = true;
    std::string asp_file = "";

    options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("test", "run unit tests")
        ("problem", value<string>(&problem), "problem to run")
        ("outputfile", value<string>(&outputfile)->default_value("output.txt"), "summary output file")
        ("policy", value<string>(&policy), "policy file (explicit POMDPs only)")
        ("size", value<int>(&size), "size of problem (problem specific)")
        ("number", value<int>(&number), "number of elements in problem (problem specific)")
        ("ghosts", value<int>(&ghosts), "number of ghosts (pocman only)")
        ("timeout", value<double>(&expParams.TimeOut), "timeout (seconds)")
        ("mindoubles", value<int>(&expParams.MinDoubles), "minimum power of two simulations")
        ("maxdoubles",value<int>(&expParams.MaxDoubles), "maximum power of two simulations")
        ("runs", value<int>(&expParams.NumRuns), "number of runs")
        ("accuracy", value<double>(&expParams.Accuracy), "accuracy level used to determine horizon")
        ("horizon", value<int>(&expParams.UndiscountedHorizon), "horizon to use when not discounting")
        ("numsteps", value<int>(&expParams.NumSteps), "number of steps to run when using average reward")
        ("verbose", value<int>(&searchParams.Verbose), "verbosity level")
        ("autoexploration", value<bool>(&expParams.AutoExploration), "Automatically assign UCB exploration constant")
        ("exploration", value<double>(&searchParams.ExplorationConstant), "Manual value for UCB exploration constant")
        ("usetransforms", value<bool>(&searchParams.UseTransforms), "Use transforms")
        ("transformdoubles", value<int>(&expParams.TransformDoubles), "Relative power of two for transforms compared to simulations")
        ("transformattempts", value<int>(&expParams.TransformAttempts), "Number of attempts for each transform")
        ("userave", value<bool>(&searchParams.UseRave), "RAVE")
        ("ravediscount", value<double>(&searchParams.RaveDiscount), "RAVE discount factor")
        ("raveconstant", value<double>(&searchParams.RaveConstant), "RAVE bias constant")
        ("treeknowledge", value<int>(&knowledge.TreeLevel), "Knowledge level in tree (0=Pure, 1=Legal, 2=Smart, 3=Rules)")
        ("rolloutknowledge", value<int>(&knowledge.RolloutLevel), "Knowledge level in rollouts (0=Pure, 1=Legal, 2=Smart, 3=Rules)")
        ("smarttreecount", value<int>(&knowledge.SmartTreeCount), "Prior count for preferred actions during smart tree search")
        ("dinamytreecount", value<bool>(&knowledge.dynamic_tree_count), "Prior count for preferred actions during smart tree search")
        ("smarttreevalue", value<double>(&knowledge.SmartTreeValue), "Prior value for preferred actions during smart tree search")
        ("disabletree", value<bool>(&searchParams.DisableTree), "Use 1-ply rollout action selection")
        ("seed", value<int>(&random_seed), "set random seed (-1 to initialize using time, >= 0 to use a fixed integer as the initial seed)")
        ("soft", value<bool>(&knowledge.soft)->default_value(true), "Use soft policy bias or hard constraints based on rules (default true, SOFT)")
        ("val", value<int>(&knowledge.val)->default_value(100), "Use rules generated from val percentage of examples (100 by default)")
        ("setW", value<double>(&W), "Fix the reward range (testing purpouse)")
        ("xes", value<bool>(&xes_log)->default_value(true), "Enable XES log");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help") != 0) {
        cout << desc << "\n";
        return 1;
    }

    if (vm.count("problem") == 0) {
        cout << "No problem specified" << endl;
        return 1;
    }

    if (vm.count("test") != 0) {
        cout << "Running unit tests" << endl;
        UnitTests();
        return 0;
    }

    SIMULATOR* real = nullptr;
    SIMULATOR* simulator = nullptr;

    XES::init(xes_log, "log.xes");

    ROCKUPDATER upd(size, number);
    if (problem == "rocksample") {
        real = new ROCKSAMPLE(size, number);
        simulator = new ROCKSAMPLE(size, number);
    } else if (problem == "random_rocksample") {
        real = new RANDOM_ROCKSAMPLE(size, number);
        simulator = new RANDOM_ROCKSAMPLE(size, number);
        upd.set_sims((RANDOM_ROCKSAMPLE *)real,(RANDOM_ROCKSAMPLE *)simulator);
        dynamic_cast<RANDOM_ROCKSAMPLE*>(real)->set_updater(&upd);
        // upd.set_simulators(&(*real), &(*simulator));
    } else {
        cout << "Unknown problem" << endl;
        exit(1);
    }

    if (vm.count("setW") != 0) {
        real->SetRewardRange(W);
        simulator->SetRewardRange(W);
    }

    if (vm.count("complexshield") != 0) {
        simulator->set_complex_shield(true);
    }

    simulator->SetKnowledge(knowledge);

    EXPERIMENT experiment(*real, *simulator, outputfile, expParams,
                          searchParams);


    if (vm.count("random_seed") != 0) {
        experiment.set_fixed_seed(random_seed);
    }

    experiment.DiscountedReturn();

    return 0;
}
