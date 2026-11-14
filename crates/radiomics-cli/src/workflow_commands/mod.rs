//! Native command adapters. Numerical fitting, prediction and evaluation live in core.
mod common;
mod predict;
mod train;
pub use predict::{predict, PredictCommand};
pub use train::{train, TrainCommand};
