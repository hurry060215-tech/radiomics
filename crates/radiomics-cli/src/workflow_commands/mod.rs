//! Native command adapters. Numerical fitting, prediction and evaluation live in core.
mod common;
mod evaluate;
mod fuse;
mod pathology;
mod predict;
mod train;
pub use evaluate::{evaluate, EvaluateCommand};
pub use fuse::{fuse, FuseCommand};
pub use pathology::{pathology, PathologyCommand};
pub use predict::{predict, PredictCommand};
pub use train::{train, TrainCommand};
