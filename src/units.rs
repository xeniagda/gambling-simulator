#![allow(non_camel_case_types)]

/// Type-level definitions of units
pub trait Unit: Copy + Send + std::fmt::Debug {
    /// Latex-formattable
    const NAME: &'static str;
    /// Value of the unit in SI
    const IN_SI: f64;

    fn from_si(val: f64) -> f64 {
        val / Self::IN_SI
    }

    fn to_si(val: f64) -> f64 {
        val * Self::IN_SI
    }
}

use crate::consts::EV_TO_J;

#[derive(Clone, Copy, Debug)]
pub struct EV;
impl Unit for EV {
    const NAME: &'static str = "eV";

    const IN_SI: f64 = EV_TO_J;
}

#[derive(Clone, Copy, Debug)]
pub struct MEV;
impl Unit for MEV {
    const NAME: &'static str = "meV";

    const IN_SI: f64 = EV_TO_J * 1e-3;
}

#[derive(Clone, Copy, Debug)]
pub struct KV_PER_CM;
impl Unit for KV_PER_CM {
    const NAME: &'static str = "kV/cm";

    const IN_SI: f64 = 1e5;
}

// why is this used so much in litterature?
#[derive(Clone, Copy, Debug)]
pub struct MILLION_CM_PER_SECOND;
impl Unit for MILLION_CM_PER_SECOND {
    const NAME: &'static str = "$10^6$ cm/s";

    const IN_SI: f64 = 1e4;
}

#[derive(Clone, Copy, Debug)]
pub struct CM_SQUARED_PER_VOLT_SECOND;
impl Unit for CM_SQUARED_PER_VOLT_SECOND {
    const NAME: &'static str = "cm^2 / V s";

    const IN_SI: f64 = 1e-4;
}

#[derive(Clone, Copy, Debug)]
pub struct PS;
impl Unit for PS {
    const NAME: &'static str = "ps";

    const IN_SI: f64 = 1e-12;
}
