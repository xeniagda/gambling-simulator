#![allow(non_camel_case_types)]

/// Type-level definitions of units
pub trait Unit: Copy + Send + std::fmt::Debug {
    /// Latex formattable
    const NAME: &'static str;

    /// Non-latex formattable
    const NAME_RAW: &'static str;

    /// Value of the unit in SI
    const IN_SI: f64;

    fn from_si(val: f64) -> f64 {
        val / Self::IN_SI
    }

    fn to_si(val: f64) -> f64 {
        val * Self::IN_SI
    }

    fn format_latex(val_si: f64) -> String {
        let val_in_unit = Self::from_si(val_si);
        format!(r"${:.4}\ [\text{{{}}}]$", val_in_unit, Self::NAME)
    }

    fn format(val_si: f64) -> String {
        let val_in_unit = Self::from_si(val_si);
        format!("{:.4} {}", val_in_unit, Self::NAME_RAW)
    }
}

macro_rules! make_unit {
    ($name:ident, $latex_name:expr, $raw_name:expr, $value:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name;
        impl Unit for $name {
            const NAME: &'static str = $latex_name;
            const NAME_RAW: &'static str = $raw_name;

            const IN_SI: f64 = $value;
        }
    }
}

make_unit!(PS, "ps", "ps", 1e-12);
make_unit!(NM, "nm", "nm", 1e-9);
make_unit!(UM2, r"um\(^2\)", "um²", 1e-12);
make_unit!(VOLT, "V", "V", 1.);
make_unit!(MILLIVOLT, "mV", "mV", 1e-3);
make_unit!(MILLIAMP, r"mA", "mA", 1e-3);
make_unit!(MICROAMP, r"\mu A", "μA", 1e-6);
make_unit!(OHM, r"\Omega", "Ω", 1.);
make_unit!(KOHM, r"k\Omega", "kΩ", 1e3);
make_unit!(THZ, "THz", "THz", 1e12);

const EV_TO_J: f64 = 1.602176e-19;
make_unit!(EV, "eV", "eV", EV_TO_J);
make_unit!(MEV, "meV", "meV", EV_TO_J * 1e-3);

make_unit!(A_PER_CM2, r"A/cm\(^2\)", "A/cm²", 1e5);
make_unit!(KV_PER_CM, "kV/cm", "kV/cm", 1e5);
make_unit!(CM_SQUARED, r"cm\(^2\)", "cm²", 1e-4);
make_unit!(MILLION_CM_PER_SECOND, r"\(10^6\) cm/s", "10^6 cm/s", 1e4);
make_unit!(CM_SQUARED_PER_SECOND, r"cm\(^2\) / s", "cm² / s", 1e-4);
make_unit!(CM_SQUARED_PER_VOLT_SECOND, r"cm\(^2\) / V s", "cm² / V s", 1e-4);

make_unit!(PER_CM_CUBED, r"cm\(^{-3}\)", "cm⁻³", 1e6);
make_unit!(ELECTRONS_PER_CM_CUBED, r"e\(^-\)/cm\(^3\)", "e⁻/cm³", crate::consts::ELECTRON_CHARGE * 1e6);

#[derive(Clone, Copy, Debug)]
pub struct DBV2PerHz;
impl Unit for DBV2PerHz {
    const NAME: &'static str = r"dB V\(^2\)/Hz";

    const NAME_RAW: &'static str = "dB V²/Hz";

    const IN_SI: f64 = 1.; // TODO: this might break some stuff.

    fn from_si(val: f64) -> f64 {
        10. * val.log10()
    }

    fn to_si(val: f64) -> f64 {
        10f64.powf(val / 10.)
    }
}
