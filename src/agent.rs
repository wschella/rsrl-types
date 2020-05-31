use crate::domains;
use crate::memory::Shared;

use rand::Rng;

pub trait OnlineLearner<S, A> {
  /// Handle a single transition collected from the problem environment.
  fn handle_transition(&mut self, transition: &domains::Transition<S, A>);

  /// Perform housekeeping after terminal state observation.
  fn handle_terminal(&mut self) {}
}

impl<S, A, T: OnlineLearner<S, A>> OnlineLearner<S, A> for Shared<T> {
  fn handle_transition(&mut self, transition: &domains::Transition<S, A>) {
    self.borrow_mut().handle_transition(transition)
  }

  fn handle_terminal(&mut self) {
    self.borrow_mut().handle_terminal()
  }
}

pub trait BatchLearner<S, A> {
  /// Handle a batch of samples collected from the problem environment.
  fn handle_batch(&mut self, batch: &[domains::Transition<S, A>]);
}

impl<S, A, T: BatchLearner<S, A>> BatchLearner<S, A> for Shared<T> {
  fn handle_batch(&mut self, batch: &[domains::Transition<S, A>]) {
    self.borrow_mut().handle_batch(batch)
  }
}

pub trait Controller<S, A> {
  /// Sample the target policy for a given state `s`.
  fn sample_target(&self, rng: &mut impl Rng, s: &S) -> A;

  /// Sample the behaviour policy for a given state `s`.
  fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> A;
}

impl<S, A, T: Controller<S, A>> Controller<S, A> for Shared<T> {
  fn sample_target(&self, rng: &mut impl Rng, s: &S) -> A {
    self.borrow().sample_target(rng, s)
  }

  fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> A {
    self.borrow().sample_behaviour(rng, s)
  }
}

pub trait OnlineController<S, A> {
    /// Sample the target policy for a given state `s`.
    fn sample_target(&mut self, rng: &mut impl Rng, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn sample_behaviour(&mut self, rng: &mut impl Rng, s: &S) -> A;

    /// Episode end
    fn handle_terminal(&mut self);
}

impl<O, A, C> OnlineController<O, A> for C
where
    C: Controller<O, A>,
{
    fn sample_target(&mut self, rng: &mut impl Rng, s: &O) -> A {
        Controller::sample_target(self, rng, s)
    }

    fn sample_behaviour(&mut self, rng: &mut impl Rng, s: &O) -> A {
        Controller::sample_behaviour(self, rng, s)
    }

    fn handle_terminal(&mut self) {}
}
