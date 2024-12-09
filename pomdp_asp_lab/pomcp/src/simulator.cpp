#include "simulator.h"

using namespace std;
std::mt19937 gen(std::random_device{}());
using namespace UTILS;

SIMULATOR::KNOWLEDGE::KNOWLEDGE()
:   TreeLevel(LEGAL),
    RolloutLevel(LEGAL),
    SmartTreeCount(10),
    dynamic_tree_count(false),
    SmartTreeValue(1.0)
{
}

SIMULATOR::STATUS::STATUS()
:   Phase(TREE),
    Particles(CONSISTENT)
{
}

SIMULATOR::SIMULATOR() 
:   Discount(1.0),
    NumActions(0),
    NumObservations(0),
    RewardRange(1.0)
{

}

SIMULATOR::SIMULATOR(int numActions, observation_t numObservations, double discount)
:   NumActions(numActions),
    NumObservations(numObservations),
    Discount(discount)
{ 
    assert(discount > 0 && discount <= 1);
}

SIMULATOR::~SIMULATOR() 
{ 
}

void SIMULATOR::Validate(const STATE& state) const 
{ 
}

bool SIMULATOR::LocalMove(STATE& state, const HISTORY& history,
    observation_t stepObs, const STATUS& status) const
{
    return true;
}

void SIMULATOR::GenerateLegal(const STATE& state, const HISTORY& history, 
    std::vector<int>& actions, const STATUS& status) const
{
    for (int a = 0; a < NumActions; ++a)
        actions.push_back(a);
}

void SIMULATOR::GeneratePreferred(const STATE& state, const HISTORY& history, 
    std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRules11(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRules15(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRules15_60(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRules15_40(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRules15_20(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

int SIMULATOR::SelectRandom(const STATE &state, const HISTORY &history,
                            const BELIEF_STATE &belief,
                            const STATUS &status) const {
    static vector<int> actions;

    if (Knowledge.RolloutLevel >= KNOWLEDGE::RULES_WEIGHT)
    {
        chrono::steady_clock::time_point start = chrono::steady_clock::now();
        
        actions.clear();
        static vector<int> actions_r;
        std::vector<double> final_prob_actions;
        actions_r.clear();
        final_prob_actions.clear();
        GenerateLegal(state, history, actions, status);
        GenerateFromRules15(state, belief, actions_r, status);
        // now we have arrays (std::vector) actions (all possible actions) and actions_r (actions suggested by rules)  

        // TODO: select actions based on a weighted probability distribution
        // 1. actions_r with probability 0.8 ($\rho_h$), actions with probability 0.2 ($1 - \rho_h$)
        // 2. actions_r is a subset of actions.
        // 3. for each action in actions, assign the correct probability and push back into final_prob_actions.
        // 4. Normalize final_prob_actions and sample from actions according to weights in final_prob_actions.

        if(actions.size() == 0) 
        {
            return 0;
        }
        else 
        {
            return actions[Random(actions.size())];
        }
    }

    if (Knowledge.RolloutLevel >= KNOWLEDGE::SMART)
    {
        actions.clear();
        GeneratePreferred(state, history, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }
        
    if (Knowledge.RolloutLevel >= KNOWLEDGE::LEGAL)
    {
        actions.clear();
        GenerateLegal(state, history, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }

    return Random(NumActions);
}

void SIMULATOR::Prior(const STATE* state, const HISTORY& history,
    VNODE* vnode, const STATUS& status) const
{
    static vector<int> actions;
    
    if (Knowledge.TreeLevel == KNOWLEDGE::PURE || state == nullptr)
    {
        vnode->SetChildren(0, 0);
        return;
    }
    else
    {
        if (Knowledge.TreeLevel <= KNOWLEDGE::SMART) {
            vnode->SetChildren(+LargeInteger, -Infinity);

            if (Knowledge.TreeLevel >= KNOWLEDGE::LEGAL) {
                actions.clear();
                GenerateLegal(*state, history, actions, status);

                for (vector<int>::const_iterator i_action = actions.begin();
                     i_action != actions.end(); ++i_action) {
                    int a = *i_action;
                    QNODE &qnode = vnode->Child(a);
                    qnode.Value.Set(0, 0);
                    qnode.AMAF.Set(0, 0);
                }
            }

            if (Knowledge.TreeLevel >= KNOWLEDGE::SMART) {
                actions.clear();
                GeneratePreferred(*state, history, actions, status);

                for (vector<int>::const_iterator i_action = actions.begin();
                     i_action != actions.end(); ++i_action) {
                    int a = *i_action;
                    QNODE &qnode = vnode->Child(a);
                    qnode.Value.Set(Knowledge.SmartTreeCount,
                                    Knowledge.SmartTreeValue);
                    qnode.AMAF.Set(Knowledge.SmartTreeCount,
                                   Knowledge.SmartTreeValue);
                }
            }
        }

        else if (Knowledge.TreeLevel == KNOWLEDGE::RULES)
        {
            chrono::steady_clock::time_point start = chrono::steady_clock::now();
            vnode->SetChildren(0, 0);

            actions.clear();
            if (Knowledge.val == 20) GenerateFromRules15_20(*state,  vnode->Beliefs(), actions, status);
            else if (Knowledge.val == 40) GenerateFromRules15_40(*state, vnode->Beliefs(), actions, status);
            else if (Knowledge.val == 60) GenerateFromRules15_60(*state, vnode->Beliefs(), actions, status);
            else GenerateFromRules15(*state, vnode->Beliefs(), actions, status);

            for (auto a : actions) {
                QNODE& qnode = vnode->Child(a);
                if (!Knowledge.dynamic_tree_count) {
                    qnode.Value.Set(Knowledge.SmartTreeCount, Knowledge.SmartTreeValue);
                    qnode.AMAF.Set(Knowledge.SmartTreeCount, Knowledge.SmartTreeValue);
                }
                else {
                    qnode.Value.Set(Knowledge.SmartTreeCount, get_tree_count(a));
                    qnode.AMAF.Set(Knowledge.SmartTreeCount, get_tree_count(a));
                }
            }

            actions.clear();
            GenerateLegal(*state, history, actions, status);

            for (int a = 0; a < NumActions; a++) {
                if (find(actions.begin(), actions.end(), a) == actions.end()) {
                    QNODE &qnode = vnode->Child(a);
                    qnode.Value.Set(+LargeInteger, -Infinity);
                    qnode.AMAF.Set(+LargeInteger, -Infinity);
                }
            }

            chrono::steady_clock::time_point end = chrono::steady_clock::now();
        }
    }
}

bool SIMULATOR::HasAlpha() const
{
    return false;
}

void SIMULATOR::AlphaValue(const QNODE& qnode, double& q, int& n) const
{
}

void SIMULATOR::UpdateAlpha(QNODE& qnode, const STATE& state) const
{
}

void SIMULATOR::DisplayBeliefs(const BELIEF_STATE& beliefState, 
    ostream& ostr) const
{
}

void SIMULATOR::DisplayState(const STATE& state, ostream& ostr) const 
{
}

void SIMULATOR::DisplayAction(int action, ostream& ostr) const 
{
    ostr << "Action " << action << endl;
}

void SIMULATOR::DisplayObservation(const STATE& state, observation_t observation, ostream& ostr) const
{
    ostr << "Observation " << observation << endl;
}

void SIMULATOR::DisplayReward(double reward, std::ostream& ostr) const
{
    ostr << "Reward " << reward << endl;
}

double SIMULATOR::GetHorizon(double accuracy, int undiscountedHorizon) const 
{ 
    if (Discount == 1)
        return undiscountedHorizon;
    return log(accuracy) / log(Discount);
}
