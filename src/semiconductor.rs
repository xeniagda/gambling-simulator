use std::f64::consts::PI;
use crate::consts::*;

/// Valley in the conduction band
/// All valleys are currently considered spherical
#[derive(Debug)]
pub struct Valley {
    pub name: &'static str,
    /// The k-vector for each equivalent minimum-point of the valley
    pub k_center: Box<[[f64; 3]]>,
    /// in terms of m0
    pub effective_mass_to_m0: f64,
    /// above gamma point valence band
    pub energy: f64,
    /// eV^-1. alpha in [monte-carlo-book-1989], kappa in [mixers-and-multipliers-2014]
    pub nonparabolicity: f64,

    /// Energy of an optical phonon at zero wave number, SI
    pub optical_phonon_energy: f64,

    /// Index x = optical phonon energy to scatter to valley index x
    pub intervalley_phonon_energy: Box<[f64]>,
}

impl Valley {
    pub fn effective_mass(&self) -> f64 {
        self.effective_mass_to_m0 * ELECTRON_MASS
    }
}

#[derive(Debug)]
pub struct Semiconductor {
    pub valleys: Box<[Valley]>,
    // in K
    pub temperature: f64,

    /// in meters
    pub lattice_constant: f64,
    /// kg/m^3
    pub density: f64,
    /// m/s
    pub sound_velocity: f64,
    /// relative dielectric constant at ω=0
    pub dielectric_static: f64,
    /// relative dielectric constant as ω → ∞
    pub dielectric_hf: f64,
    /// in J
    pub acoustic_deformation_potential: f64,
    /// in J/m
    pub intervalley_deformation_potential: f64,
}

impl Semiconductor {
    pub fn GaAs(temperature: f64) -> Semiconductor {
        let a = 5.65e-10;
        // TODO: Cite values here. Currently taken from Helena's copy with some modifications
        // Optical phonon energy is taken from [multipliers-and-mixers-2014]
        let Γ = Valley {
            name: "Γ",
            k_center: vec![[0., 0., 0.]].into(),
            effective_mass_to_m0: 0.067,
            energy: 1.42 * EV_TO_J,
            nonparabolicity: 0.61 / EV_TO_J,
            optical_phonon_energy: 36.13e-3 * EV_TO_J,
            intervalley_phonon_energy: vec![27.8e-3 * EV_TO_J, 27.8e-3 * EV_TO_J, 29.3e-3 * EV_TO_J].into_boxed_slice(),
        };
        let L_xyz = 1.61993 / a;
        let L = Valley {
            name: "L",
            k_center: vec![[L_xyz, L_xyz, L_xyz,], [-L_xyz, L_xyz, L_xyz,], [L_xyz, -L_xyz, L_xyz,], [L_xyz, L_xyz, -L_xyz,]].into(),
            effective_mass_to_m0: 0.3,
            energy: 1.71 * EV_TO_J,
            nonparabolicity: 0.222 / EV_TO_J,
            optical_phonon_energy: 36.13e-3 * EV_TO_J,
            intervalley_phonon_energy: vec![27.8e-3 * EV_TO_J, 27.8e-3 * EV_TO_J, 29.3e-3 * EV_TO_J].into_boxed_slice(),
        };
        // TODO: X is not quite the right name for this valley. The valley on the Δ-line (from Γ to X) but not all the way to X.
        // In [monte-carlo-book-1989] they call this valley the Δ-valley, in [multipliers-and-mixers-2014] they call it the X valley.
        // [sc-data-handbook] says it's about 10% from the real X point. so let's go with that
        let X_xyz = 3.1415 / a * 0.9;
        let X = Valley {
            name: "X",
            k_center: vec![[X_xyz, 0., 0.,], [0., X_xyz, 0.,], [0., 0., X_xyz,]].into(),
            effective_mass_to_m0: 0.85,
            energy: 1.90 * EV_TO_J,
            nonparabolicity: 0.061 / EV_TO_J,
            optical_phonon_energy: 36.13e-3 * EV_TO_J,
            intervalley_phonon_energy: vec![29.3e-3 * EV_TO_J, 29.3e-3 * EV_TO_J, 29.3e-3 * EV_TO_J].into_boxed_slice(),
        };
        Semiconductor {
            valleys: vec![Γ, L, X].into(),
            temperature,
            lattice_constant: a,
            density: 5.37 * 1000.,
            sound_velocity: 5220.,
            dielectric_static: 12.53,
            dielectric_hf: 10.82,
            acoustic_deformation_potential: 7.0 * EV_TO_J,
            intervalley_deformation_potential: 1.0e9 * EV_TO_J * 100.,
        }
    }
}

pub struct Electron<'sc> {
    /// Semiconductor we are in
    pub sc: &'sc Semiconductor,
    /// self.sc[self.valley_idx]
    /// TODO: Should we need to track which equivalent valley we are in?
    pub valley_idx: usize,
    /// k-vector relative to the center of the valley we are in
    /// m^-1
    pub k: [f64; 3],
}

impl Electron<'_> {
    pub fn valley(&self) -> &Valley {
        &self.sc.valleys[self.valley_idx]
    }

    pub fn k_mag2(&self) -> f64 {
        self.k.iter().map(|x| x.powi(2)).sum()
    }

    pub fn k_mag(&self) -> f64 {
        self.k_mag2().sqrt()
    }

    // in J, relative to valley center
    pub fn energy(&self) -> f64 {
        let valley = self.valley();
        // all local variables in SI units
        let spherical_energy = PLANCK_SI.powi(2) * self.k_mag2() / (2. * valley.effective_mass());
        (-1. + (1. + 4. * spherical_energy * valley.nonparabolicity).sqrt()) / (2. * valley.nonparabolicity)
    }

    pub fn energy_eV(&self) -> f64 {
        self.energy() * J_TO_EV
    }

    // in m/s
    pub fn velocity(&self) -> f64 {
        let energy = self.energy();
        PLANCK_SI * self.k_mag() / (self.valley().effective_mass() * (1. + 2. * self.valley().nonparabolicity * energy))
    }
}

#[derive(PartialEq, Clone, Copy)]
pub enum PhononType { Emission, Absorption }

// All rates in s^-1
// TODO: We should return some extra information about E' and distribution of k' ?
impl Electron<'_> {
    // Phonon energy assumed small, independent of emission/absorption
    // Indep. of k', E' = E
    // TODO: Slightly different form in [monte-carlo-transport-gaas-1969]
    pub fn scattering_rate_intravalley_acoustic_phonon(&self, _ty: PhononType) -> f64 {
        let valley = self.valley();
        let E = self.energy();
        2f64.sqrt() * valley.effective_mass().powf(1.5) * BOLTZMANN * self.sc.temperature * self.sc.acoustic_deformation_potential.powi(2)
            / (PI * PLANCK_SI.powi(4) * self.sc.sound_velocity.powi(2) * self.sc.density)
            * E.sqrt() * (1. + 2. * E * valley.nonparabolicity) * (1. + E * valley.nonparabolicity).sqrt()
    }

    // Dependent on 1/|k - k'|^2
    // E' = E ± E_phonon
    // From monte-carlo-transport-gaas-1969
    pub fn scattering_rate_intravalley_optical_phonon(&self, ty: PhononType) -> f64 {
        let valley = self.valley();

        let α = valley.nonparabolicity;
        let N_op = 1./(-1. + (valley.optical_phonon_energy / (BOLTZMANN * self.sc.temperature)).exp());

        let N_op_eff = if ty == PhononType::Emission { N_op + 1. } else { N_op };
        let E = self.energy();
        let E_ = if ty == PhononType::Emission { E - valley.optical_phonon_energy } else { E + valley.optical_phonon_energy };
        let γE = E * (1. + α * E);
        let γE_ = E_ * (1. + α * E_);

        // can't scatter into if we have too low energy
        if E_ <= 0. {
            return 0.;
        }

        let A = (2. * (1. + α * E) * (1. + α * E_) + α * (γE + γE_)).powi(2);
        let B = -2. * α * γE.sqrt() * γE_.sqrt() * (4. * (1. + α * E) * (1. + α * E_) + α * (γE + γE_));
        let C = 4. * (1. + α * E) * (1. + α * E_) * (1. + 2. * α * E) * (1. + 2. * α * E_);
        let F0 = (A * ((γE.sqrt() + γE_.sqrt()) / (γE.sqrt() - γE_.sqrt())).abs().ln() + B) / C;


        N_op_eff * ELECTRON_CHARGE.powi(2) * valley.effective_mass().sqrt() * valley.optical_phonon_energy
            / (2f64.sqrt() * PLANCK_SI.powi(2) * (4. * PI * EPS0))
            * (1. / self.sc.dielectric_hf - 1. / self.sc.dielectric_static)
            * (1. + 2. * α * E_) / γE.sqrt()
            * F0
    }

    // Isotropic
    // E' = E ± E_phonon - E_fi
    pub fn scattering_rate_intervalley_optical_phonon(&self, ty: PhononType, destination_valley_idx: usize) -> f64 {
        let this_valley = self.valley();
        let dest_valley = &self.sc.valleys[destination_valley_idx];

        let n_dest_valleys = dest_valley.k_center.len();
        // can't scatter into ourselves bozo
        let n_dest_valleys = if destination_valley_idx == self.valley_idx { n_dest_valleys - 1 } else { n_dest_valleys };

        let phonon_energy = this_valley.intervalley_phonon_energy[destination_valley_idx];

        let N_op = 1./(-1. + (this_valley.optical_phonon_energy / (BOLTZMANN * self.sc.temperature)).exp());
        let N_op_eff = if ty == PhononType::Emission { N_op + 1. } else { N_op };

        let E = self.energy();
        let E_fi = dest_valley.energy - this_valley.energy;
        let E_ = if ty == PhononType::Emission { E - phonon_energy - E_fi } else { E + phonon_energy - E_fi };

        N_op_eff * n_dest_valleys as f64 * this_valley.effective_mass().powf(1.5) * self.sc.intervalley_deformation_potential.powi(2) * E_.sqrt()
            / (2f64.sqrt() * PI * self.sc.density * PLANCK_SI.powi(2) * phonon_energy)
    }

}

