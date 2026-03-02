#![allow(non_snake_case, mixed_script_confusables, confusable_idents)] // for band names such as Γ and L etc

pub mod consts;
pub mod semiconductor;
pub mod units;
pub mod histogram;

/// Utility function to guarantee a struct is Send
pub fn ensure_send<T: Send>() {}

/// Utility function to guarantee a struct is Sync
pub fn ensure_sync<T: Sync>() {}
