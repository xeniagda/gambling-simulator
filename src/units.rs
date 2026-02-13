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

macro_rules! make_unit {
    ($name:ident, $pretty_name:expr, $value:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name;
        impl Unit for $name {
            const NAME: &'static str = $pretty_name;

            const IN_SI: f64 = $value;
        }
    }
}

make_unit!(PS, "ps", 1e-12);
make_unit!(NM, "nm", 1e-9);

const EV_TO_J: f64 = 1.602176e-19;
make_unit!(EV, "eV", EV_TO_J);
make_unit!(MEV, "meV", EV_TO_J * 1e-3);

make_unit!(KV_PER_CM, "kV/cm", 1e5);
make_unit!(MILLION_CM_PER_SECOND, "$10^6$ cm/s", 1e4);
make_unit!(CM_SQUARED_PER_VOLT_SECOND, "cm$^2$ / V s", 1e-4);

make_unit!(PER_CM_CUBED, "cm$^{-3}$", 1e-6);
