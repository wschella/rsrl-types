//! Traits and structs for implementing Reinforcement Learning environments
//! and agents.
//!
//! Copied from <https://github.com/tspooner/rsrl>

extern crate rand;

pub mod agent;
pub mod domains;
pub mod memory;

pub use agent::*;
pub use domains::*;
pub use memory::*;

pub extern crate spaces;
