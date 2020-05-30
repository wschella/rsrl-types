extern crate rand;
extern crate spaces;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use spaces::Space;

extern crate rsrl_types;

use rsrl_types::OnlineLearner;
use rsrl_types::{Action, Controller, Domain, Observation, State, Transition};

fn main() {
    let rng = &mut rand::thread_rng();
    let mut env = DummyDomain::new();
    let mut agent = RandomAgent::new(env.action_space().card().into());

    let n_steps = 10;

    // Sample the initial action
    let mut action = agent.sample_target(rng, &env.emit());

    for step in 0..n_steps + 1 {
        // Step the environment
        let transition = env.step(action);

        // Let the agent learn from the transition
        agent.handle_transition(&transition);

        // Check wether episode ended
        if transition.to.is_terminal() {
            println!(
                "Step {} reached the end with transition {:?}",
                step, transition
            );
            break;
        }

        // Let the agent take an action
        action = agent.sample_behaviour(rng, &transition.to);

        println!(
            "Step {} transition {:?} action {}",
            step, transition, action
        );
    }
}

/// Dummy domain that is a simple line with 2 actions, walking back and forward.
/// Reaching the end of the line terminates the episode.
struct DummyDomain {
    state: i64,
}

impl DummyDomain {
    pub fn new() -> Self {
        DummyDomain { state: 0 }
    }

    fn update_state(&mut self, action: usize) -> f64 {
        if action == 0 {
            self.state = std::cmp::max(self.state - 1, 0);
        } else {
            self.state = std::cmp::min(self.state + 1, 4);
        }

        if self.state == 4 {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    fn is_terminal(&self) -> bool {
        match self.state {
            4 => true,
            _ => false,
        }
    }
}

impl Domain for DummyDomain {
    type StateSpace = spaces::discrete::Interval;
    type ActionSpace = spaces::discrete::Ordinal;

    fn emit(&self) -> Observation<State<Self>> {
        match self.is_terminal() {
            true => Observation::Terminal(self.state),
            false => Observation::Full(self.state),
        }
    }

    fn step(&mut self, action: Action<Self>) -> Transition<State<Self>, Action<Self>> {
        let from = self.emit();
        let reward = self.update_state(action);
        Transition {
            from,
            action,
            reward,
            to: self.emit(),
        }
    }

    fn state_space(&self) -> Self::StateSpace {
        spaces::discrete::Interval::new(Some(0), Some(4))
    }

    fn action_space(&self) -> Self::ActionSpace {
        spaces::discrete::Ordinal::new(2usize)
    }
}

pub struct RandomAgent {
    pub action_count: usize,
    policy: Uniform<usize>,
}

impl RandomAgent {
    pub fn new(action_count: usize) -> Self {
        RandomAgent {
            action_count,
            policy: Uniform::new(0, action_count),
        }
    }
}

impl<S> Controller<S, usize> for RandomAgent {
    fn sample_target(&self, rng: &mut impl Rng, _s: &S) -> usize {
        self.policy.sample(rng)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, _s: &S) -> usize {
        self.policy.sample(rng)
    }
}

impl<S, A> OnlineLearner<S, A> for RandomAgent {
    fn handle_transition(&mut self, _transition: &Transition<S, A>) {
        // NOP
    }
}
