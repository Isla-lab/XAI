#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "history.h"
#include "node.h"
#include "XES_logger.h"
#include "utils.h"
#include <iostream>
#include <math.h>
#include <random>
#include <stdexcept>

struct BOUNDS {
    double x1_loose;
    double x1_strict;
    double x2_loose;
    double x2_strict;
    double x3_loose;
    double x3_strict;
    double x4_loose;
    double x4_strict;
};


class BELIEF_STATE;

class STATE : public MEMORY_OBJECT
{
};

class SIMULATOR
{
public:
    using observation_t = uint64_t;

    struct ActionList{
        std::vector<int> actions;
        std::vector<int> times;
    };

    struct KNOWLEDGE
    {
        enum
        {
            PURE,
            LEGAL,
            SMART,
            RULES,
            RULES_WEIGHT,
            NUM_LEVELS
        };

        KNOWLEDGE();
        
        int RolloutLevel;
        int TreeLevel;
        int SmartTreeCount;
        bool dynamic_tree_count;
        double SmartTreeValue;
        int val;
        bool soft;
        
        int Level(int phase) const
        {
            assert(phase < STATUS::NUM_PHASES);
            if (phase == STATUS::TREE)
                return TreeLevel;
            else
                return RolloutLevel;
        }
    };

    struct STATUS
    {
        STATUS();
        
        enum
        {
            TREE,
            ROLLOUT,
            NUM_PHASES
        };
        
        enum
        {
            CONSISTENT,
            INCONSISTENT,
            RESAMPLED,
            OUT_OF_PARTICLES
        };
        
        int Phase;
        int Particles;
    };

    SIMULATOR();
    SIMULATOR(int numActions, observation_t numObservations, double discount = 1.0);    
    virtual ~SIMULATOR();

    // Create start start state (can be stochastic)
    virtual STATE* CreateStartState() const = 0;

    // Free memory for state
    virtual void FreeState(STATE* state) const = 0;

    // Update state according to action, and get observation and reward. 
    // Return value of true indicates termination of episode (if episodic)
    virtual bool Step(STATE& state, int action, 
        observation_t& observation, double& reward) const = 0;

    virtual bool Step(const VNODE *const mcts_root, STATE &state, int action, 
        observation_t& observation, double& reward, double min, double max) const {
        throw std::runtime_error("unsupported");
    }


    virtual bool Step(const VNODE *const mcts_root, STATE &state, int action, 
        observation_t& observation, double& reward, const BOUNDS &s) const {
        throw std::runtime_error("unsupported");
    }
        
    virtual void pre_run() {}
    virtual void post_run() {}

    // Create new state and copy argument (must be same type)
    virtual STATE* Copy(const STATE& state) const = 0;
    
    // Sanity check
    virtual void Validate(const STATE& state) const;

    // Modify state stochastically to some related state
    virtual bool LocalMove(STATE& state, const HISTORY& history,
        observation_t stepObs, const STATUS& status) const;

    // Use domain knowledge to assign prior value and confidence to actions
    // Should only use fully observable state variables
    void Prior(const STATE* state, const HISTORY& history, VNODE* vnode,
        const STATUS& status) const;

    // Use domain knowledge to select actions stochastically during rollouts
    // Should only use fully observable state variables
    int SelectRandom(const STATE &state, const HISTORY &history,
                     const BELIEF_STATE &belief, const STATUS &status) const;

    // Generate set of legal actions
    virtual void GenerateLegal(const STATE& state, const HISTORY& history, 
        std::vector<int>& actions, const STATUS& status) const;

    // Generate set of preferred actions
    virtual void GeneratePreferred(const STATE& state, const HISTORY& history, 
        std::vector<int>& actions, const STATUS& status) const;
    
    // used to generate adaptive tree count
    virtual int get_tree_count(int action) const {
        return Knowledge.SmartTreeCount;
    }

    // Generate set of preferred actions using a rule-based approach
    virtual void GenerateFromRules15(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const;
    virtual void GenerateFromRules15_20(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const;
    virtual void GenerateFromRules15_60(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const;
    virtual void GenerateFromRules15_40(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const;
    virtual void GenerateFromRules11(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const;

    // For explicit POMDP computation only
    virtual bool HasAlpha() const;
    virtual void AlphaValue(const QNODE& qnode, double& q, int& n) const;
    virtual void UpdateAlpha(QNODE& qnode, const STATE& state) const;

    // Textual display
    virtual void DisplayBeliefs(const BELIEF_STATE& beliefState, 
        std::ostream& ostr) const;
    virtual void DisplayState(const STATE& state, std::ostream& ostr) const;
    virtual void DisplayAction(int action, std::ostream& ostr) const;
    virtual void DisplayObservation(const STATE& state, observation_t observation, std::ostream& ostr) const;
    virtual void DisplayReward(double reward, std::ostream& ostr) const;

    // xes logging
    virtual void log_problem_info() const {}
    virtual void log_run_info() const {}
    virtual void log_beliefs(const BELIEF_STATE& beliefState) const {}
    virtual void log_state(const STATE& state) const {}
    virtual void log_action(int action) const {}
    virtual void log_observation(const STATE& state, observation_t observation) const {}
    virtual void log_reward(double reward) const {}

    // Accessors
    void SetKnowledge(const KNOWLEDGE& knowledge) { Knowledge = knowledge; }
    int GetNumActions() const { return NumActions; }
    int GetNumObservations() const { return NumObservations; }
    bool IsEpisodic() const { return false; }
    double GetDiscount() const { return Discount; }
    double GetRewardRange() const { return RewardRange; }
    void SetRewardRange(double W) { RewardRange = W; }
    void NormProbActions(bool X){
        if (X){
            double sum = accumulate(prob_actions.begin(), prob_actions.end(), 0.0);
            for_each(prob_actions.begin(), prob_actions.end(), [sum](double &c){ c /= sum; });
        }
    }
    // void NormProbActualActions(){
    //     double sum = accumulate(prob_actual.begin(), prob_actual.end(), 0.0);
    //     for_each(prob_actual.begin(), prob_actual.end(), [sum](double &c){ c /= sum; });
    // }
    double GetHorizon(double accuracy, int undiscountedHorizon = 100) const;
    bool use_complex_shield() const { return complex_shield; }
    void set_complex_shield(bool b) { complex_shield = b; }

    // shielding
    virtual void pre_shield(const BELIEF_STATE &belief, std::vector<int> &legal_actions) const {
        legal_actions.clear();
        for (int action = 0; action < GetNumActions(); action++)
            legal_actions.push_back(action);
    }

    virtual void set_belief_metainfo(VNODE *v, const SIMULATOR &s) const {
        v->Beliefs().set_metainfo(BELIEF_META_INFO(), *this);
    }

    virtual void FreeMetainfo(BELIEF_META_INFO *m) const {
        delete m;
    }

    void set_seed(size_t s) const {
        random_state.seed(s);
    }

    virtual Classification check_rule(const BELIEF_META_INFO &m, int a, double t) const {
        return TRUE_POSITIVE;
    }

    void SetProbActions(std::vector<double> data){
        prob_actions = data;
    }

protected:
    int NumActions;
    observation_t NumObservations;
    bool complex_shield = false;
    double Discount, RewardRange;
    mutable std::vector<double> prob_actions;
    mutable ActionList actionList;
    mutable int num_best, num_worst = 0;
    mutable double last_time = 0;
    // std::vector<double> prob_actions = {0.649, 0.570, 0.649, 0.733, 0.2, 0.849, 0.657}; //Probabilities of actions depending of counterexamples of axioms - ADD 0.2 AT 5th POSITION FOR EXIT
    mutable std::vector<double> prob_actual; //to match with actual actions generated
    KNOWLEDGE Knowledge;
    mutable std::mt19937 random_state;

};

#endif // SIMULATOR_H
