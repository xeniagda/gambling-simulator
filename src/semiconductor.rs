use std::f64::consts::PI;
use rand::Rng;

use crate::consts::*;

/// Valley in the conduction band
/// All valleys are currently considered spherical
#[derive(Debug, Clone)]
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

    // What k vector magnitude for getting a specific energy in this valley
    pub fn kmag_for_e(&self, e: f64) -> f64 {
        let spherical_energy = e * (1. + self.nonparabolicity * e);
        (spherical_energy * 2. * self.effective_mass()).sqrt() / PLANCK_BAR_SI
    }

}

#[derive(Debug, Clone)]
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
    pub relative_dielectric_static: f64,
    /// relative dielectric constant as ω → ∞
    pub relative_dielectric_hf: f64,
    /// in J
    pub acoustic_deformation_potential: f64,
    /// in J/m
    pub intervalley_deformation_potential: f64,

    /// in m^-3
    pub impurity_density: f64,
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
            intervalley_phonon_energy: vec![0., 27.8e-3 * EV_TO_J, 29.3e-3 * EV_TO_J].into_boxed_slice(),
        };
        let L_xyz = 1.61993 / a;
        let L = Valley {
            name: "L",
            k_center: vec![[L_xyz, L_xyz, L_xyz,], [-L_xyz, L_xyz, L_xyz,], [L_xyz, -L_xyz, L_xyz,], [L_xyz, L_xyz, -L_xyz,]].into(),
            effective_mass_to_m0: 0.35,
            energy: 1.729 * EV_TO_J,
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
            energy: 1.906 * EV_TO_J,
            nonparabolicity: 0.061 / EV_TO_J,
            optical_phonon_energy: 36.13e-3 * EV_TO_J,
            intervalley_phonon_energy: vec![29.3e-3 * EV_TO_J, 29.3e-3 * EV_TO_J, 29.3e-3 * EV_TO_J].into_boxed_slice(),
        };
        Semiconductor {
            valleys: vec![Γ, L, X].into(),
            temperature,
            lattice_constant: a,
            density: 5370.,
            sound_velocity: 5220.,
            relative_dielectric_static: 12.53,
            relative_dielectric_hf: 10.82,
            acoustic_deformation_potential: 7.0 * EV_TO_J, // Value from [multipliers-and-mixers-2014]
            intervalley_deformation_potential: 1e11  * EV_TO_J, // value from [monte-carlo-review-1991], is way bigger than in [multipliers-and-mixers] ??
            impurity_density: 1.0e14 * 1e6, // 1e17 cm^-3 used in [multipliers-and-mixers-2014]
        }
    }
}

#[derive(Clone)]
pub struct Electron<'sc> {
    /// Semiconductor we are in
    pub sc: &'sc Semiconductor,
    /// self.sc[self.valley_idx]
    /// TODO: Should we need to track which equivalent valley we are in?
    pub valley_idx: usize,
    /// k-vector relative to the center of the valley we are in
    /// m^-1
    pub k: [f64; 3],
    pub pos: [f64; 3],
}

impl<'sc> Electron<'sc> {
    // electron with average temperature of the lattice
    pub fn thermalized<R: Rng>(rng: &mut R, sc: &'sc Semiconductor, valley_idx: usize, pos: [f64; 3]) -> Self {
        let valley = &sc.valleys[valley_idx];
        let var = BOLTZMANN * sc.temperature / valley.effective_mass();

        // TODO: Sample normal distribution instead of uniform
        //
        // [a, b] has var 1/12(b-a)^2
        // [-a, a] has var 1/12(2a)^2 = 1/3 a^2
        // a = 3sqrt(var)

        let distr = rand_distr::Normal::new(0., var.sqrt()).unwrap();
        let vx = rng.sample(&distr);
        let vy = rng.sample(&distr);
        let vz = rng.sample(&distr);


        let vtok = valley.effective_mass() / PLANCK_BAR_SI;

        Electron {
            sc,
            valley_idx,
            k: [vtok * vx, vtok * vy, vtok * vz],
            pos,
        }
    }

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
        let spherical_energy = PLANCK_BAR_SI.powi(2) * self.k_mag2() / (2. * valley.effective_mass());
        (-1. + (1. + 4. * spherical_energy * valley.nonparabolicity).sqrt()) / (2. * valley.nonparabolicity)
    }

    pub fn energy_eV(&self) -> f64 {
        self.energy() * J_TO_EV
    }

    /// Velocity vector, taking into account nonparabolicity
    pub fn velocity(&self) -> [f64; 3] {
        let energy = self.energy();
        let nonparab = 1. + 2. * self.valley().nonparabolicity * energy;
        let f = PLANCK_BAR_SI / self.valley().effective_mass() / nonparab;
        [f * self.k[0], f * self.k[1], f * self.k[2]]
    }
}

struct CoordSystem {
    xhat: [f64; 3],
    yhat: [f64; 3],
    zhat: [f64; 3],
}

impl CoordSystem {
    fn given_z(z: [f64; 3]) -> CoordSystem {
        let zmag = z.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        let zhat = if zmag > 0. {
            [z[0] / zmag, z[1] / zmag, z[2] / zmag]
        } else {
            // Arbitrary vector
            [1., 0., 0.]
        };

        // Define xxhat = normalize(xhat cross zzhat) or normalize(yhat cross zzhat) (depending on which is more normalizable)
        let xhat = {
            let x1 = [zhat[2], 0., -zhat[0]];
            let x1mag = x1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let x2 = [0., -zhat[0], zhat[1]];
            let x2mag = x2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let (x, xmag) = if x1mag > x2mag { (x1, x1mag) } else { (x2, x2mag) };
            [x[0] / xmag, x[1] / xmag, x[2] / xmag]
        };
        // Define yyhat = normalize(xxhat cross zzhat) = xxhat cross zzhat since both are already normalized and perpendicular
        let yhat = [
            xhat[1] * zhat[2] - xhat[2] * zhat[1],
            xhat[2] * zhat[0] - xhat[0] * zhat[2],
            xhat[0] * zhat[1] - xhat[1] * zhat[0],
        ];
        CoordSystem {
            xhat, yhat, zhat,
        }
    }

    /// theta = 0 points in zhat
    /// theta = pi/2, phi=0 points in xhat
    fn cylidrical(&self, len: f64, theta: f64, phi: f64) -> [f64; 3] {
        let v_zz = len * theta.cos();
        let v_xx = len * theta.sin() * phi.cos();
        let v_yy = len * theta.sin() * phi.sin();

        let v_x = v_zz * self.zhat[0] + v_xx * self.xhat[0] + v_yy * self.yhat[0];
        let v_y = v_zz * self.zhat[1] + v_xx * self.xhat[1] + v_yy * self.yhat[1];
        let v_z = v_zz * self.zhat[2] + v_xx * self.xhat[2] + v_yy * self.yhat[2];
        [v_x, v_y, v_z]
    }
}

pub struct ScatteringMechanism<'sc, R: Rng> {
    pub name_full: &'static str,
    pub name_short: &'static str,
    /// Scattering rate for a specific electron at it's energy
    /// Given in s^-1
    pub rate: fn(&Electron<'sc>) -> f64,
    /// Maximum rate for this electron for any energy below emax
    /// May over-estimate the rate
    pub maximum_rate:  fn(&Electron<'sc>, emax: f64) -> f64,
    // TODO: Maybe it's better to take a &mut Electron?
    // Electron doesn't carry much data though so whatever
    pub resulting_state: fn(&Electron<'sc>, &mut R) -> Electron<'sc>,
}

#[derive(PartialEq, Clone, Copy)]
pub enum PhononType { Emission, Absorption }

// All rates in s^-1
impl<'sc> Electron<'sc> {
    pub fn all_mechanisms<R: Rng>() -> Box<[ScatteringMechanism<'sc, R>]> {
        let intra_ac_phonon = ScatteringMechanism {
            name_full: "Intravalley acoustic phonon",
            name_short: "intra ac. phonon",
            rate: |e| e.rate_intra_ac_phonon(None),
            // Acoustic phonon scattering rate is strictly increasing
            maximum_rate: |el, en| el.rate_intra_ac_phonon(Some(en)),
            resulting_state: |e, r| e.scatter_isotropic(r, e.energy()),
        };
        let intra_opt_phonon_abs = ScatteringMechanism {
            name_full: "Intravalley optical phonon absorption",
            name_short: "intra opt. phonon abs.",
            rate: |e| e.rate_intra_opt_phonon(PhononType::Absorption, None),
            // Acoustic phonon scattering rate is strictly decreasing
            maximum_rate: |el, _en| el.rate_intra_opt_phonon(PhononType::Absorption, Some(0.)),
            // For absorption we gain energy from the phonon
            resulting_state: |e, r| e.scatter_mag2(r, e.energy() + e.valley().optical_phonon_energy),
        };
        let intra_opt_phonon_em = ScatteringMechanism {
            name_full: "Intravalley optical phonon emission",
            name_short: "intra opt. phonon em.",
            rate: |e| e.rate_intra_opt_phonon(PhononType::Emission, None),
            // The maximum is somewhere around 130meV
            // Add a factor of 2 for safety (:
            maximum_rate: |el, _en| 2. * el.rate_intra_opt_phonon(PhononType::Emission, Some(0.125 * EV_TO_J)),
            // For absorption we gain energy from the phonon
            resulting_state: |e, r| e.scatter_mag2(r, e.energy() - e.valley().optical_phonon_energy),
        };
        let impurity = ScatteringMechanism {
            name_full: "Impurity scattering",
            name_short: "imp.",
            rate: |e| e.rate_impurity(None),
            // Strictly increasing
            maximum_rate: |el, en| el.rate_impurity(Some(en)),
            resulting_state: |e, r| e.scatter_impurity(r),
        };
        let mut mechanisms = vec![intra_ac_phonon, intra_opt_phonon_abs, intra_opt_phonon_em, impurity];

        macro_rules! gen_intervalley {
            ($dest_valley_idx:expr, $name:expr) => {
                let inter_opt_phonon_abs = ScatteringMechanism {
                    name_full: concat!("Intervalley optical phonon absorption to ", $name),
                    name_short: concat!("inter →", $name, " opt. phonon abs"),
                    rate: |e| e.rate_inter_opt_phonon(PhononType::Absorption, $dest_valley_idx, None),
                    // Strictly increasing
                    maximum_rate: |el, en| el.rate_inter_opt_phonon(PhononType::Absorption, $dest_valley_idx, Some(en)),
                    // For absorption we gain energy from the phonon
                    resulting_state: |e, r| {
                        let energy_after = e.energy() + e.valley().energy - e.sc.valleys[$dest_valley_idx].energy + e.valley().intervalley_phonon_energy[$dest_valley_idx];
                        let mut e_ = e.clone();
                        e_.valley_idx = $dest_valley_idx;
                        e_.scatter_isotropic(r, energy_after)
                    },
                };
                mechanisms.push(inter_opt_phonon_abs);

                let inter_opt_phonon_em = ScatteringMechanism {
                    name_full: concat!("Intervalley optical phonon emission to ", $name),
                    name_short: concat!("inter →", $name, " opt. phonon em"),
                    rate: |e| e.rate_inter_opt_phonon(PhononType::Emission, $dest_valley_idx, None),
                    // Strictly increasing
                    maximum_rate: |el, en| el.rate_inter_opt_phonon(PhononType::Emission, $dest_valley_idx, Some(en)),
                    // For absorption we gain energy from the phonon
                    resulting_state: |e, r| {
                        let energy_after = e.energy() + e.valley().energy - e.sc.valleys[$dest_valley_idx].energy - e.valley().intervalley_phonon_energy[$dest_valley_idx];
                        let mut e_ = e.clone();
                        e_.valley_idx = $dest_valley_idx;
                        e_.scatter_isotropic(r, energy_after)
                    },
                };
                mechanisms.push(inter_opt_phonon_em);
            }
        }
        // only works for GaAs!!!!
        gen_intervalley!(0, "Γ");
        gen_intervalley!(1, "L");
        gen_intervalley!(2, "X");

        mechanisms.into_boxed_slice()
    }

    /// Resulting state after an isotropic (independent of k') collision resulting in an energy of `res_energy`
    /// TODO: This assumes parabolic bands for most scattering mechanisms dependent on the overlap integral between k and k'
    /// For low energy (αE << 1) G(k, k') is isotopic
    pub fn scatter_isotropic<R: Rng>(&self, rng: &mut R, res_energy: f64) -> Electron<'sc> {
        let k_res_mag = self.valley().kmag_for_e(res_energy);

        // Spherical coordinates: pick ϕ uniformly, pick θ = asin(r) for r ∈ [-1, 1]
        let phi = rng.random_range(0. ..= 2.*PI);
        let theta = rng.random_range(-1f64 ..= 1f64).acos();

        let k_res = [
            k_res_mag * phi.cos() * theta.sin(),
            k_res_mag * phi.sin() * theta.sin(),
            k_res_mag * theta.cos(),
        ];
        Electron {
            k: k_res,
            ..*self
        }
    }

    // Resulting state after a collision where the probability of a state k' is proportional to |k-k'|⁻²
    pub fn scatter_mag2<R: Rng>(&self, rng: &mut R, res_energy: f64) -> Electron<'sc> {
        let kmag = self.k.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let k_res_mag = self.valley().kmag_for_e(res_energy);

        // create a coordinate system with zz = k
        let coord_sys = CoordSystem::given_z(self.k);

        // Pick a point on the unit sphere weighted the distance to k' taken in direction zz
        let x = kmag / k_res_mag;

        let r = rng.random_range(0f64..=1f64);

        let costheta = -1./(2.*x) * (-1. - x.powi(2) + (x-1.).powi(2) * ((x+1.)/(x-1.)).abs().powf(2. * r));

        let theta = costheta.acos();
        let phi = rng.random_range(0. ..= 2.*PI);

        let k_res = coord_sys.cylidrical(k_res_mag, theta, phi);

        Electron {
            k: k_res,
            ..*self
        }
    }

    // CW approach, from [monte-carlo-book-1989.pdf]
    pub fn scatter_impurity<R: Rng>(&self, rng: &mut R) -> Electron<'sc> {
        let kmag = self.k.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let coord_sys = CoordSystem::given_z(self.k);

        let phi = rng.random_range(0. ..= 2.*PI);

        let impurity_mean_dist = (3. / (4. * PI * self.sc.impurity_density)).powf(1./3.);
        let eps_b = ELECTRON_CHARGE.powi(2) / (2. * self.sc.relative_dielectric_static * EPS0 * impurity_mean_dist);
        let r = rng.random_range(0f64..=1f64);
        let E = self.energy();
        let f = (E * (1. + self.valley().nonparabolicity * E) / (eps_b * (1. + 2. * self.valley().nonparabolicity * E))).powi(2);
        let theta = ((f * r - 1.) / (f * r + 1.)).acos();

        let k_res = coord_sys.cylidrical(kmag, theta, phi);

        Electron {
            k: k_res,
            ..*self
        }
    }

    // Phonon energy assumed small, independent of emission/absorption
    // Indep. of k', E' = E
    // Matches [monte-carlo-gaas-1980] (mod a factor of 2)
    pub fn rate_intra_ac_phonon(&self, E: Option<f64>) -> f64 {
        let valley = self.valley();
        let E = E.unwrap_or_else(|| self.energy());
        2f64.sqrt() * valley.effective_mass().powf(1.5) * BOLTZMANN * self.sc.temperature * self.sc.acoustic_deformation_potential.powi(2)
            / (PI * PLANCK_BAR_SI.powi(4) * self.sc.sound_velocity.powi(2) * self.sc.density)
            * E.sqrt() * (1. + 2. * E * valley.nonparabolicity) * (1. + E * valley.nonparabolicity).sqrt()
    }

    // Dependent on 1/|k - k'|^2
    // E' = E ± E_phonon
    // Matches [monte-carlo-gaas-1980] (plus correction factors from [monte-carlo-transport-gaas-1969])
    pub fn rate_intra_opt_phonon(&self, ty: PhononType, E: Option<f64>) -> f64 {
        let valley = self.valley();

        let α = valley.nonparabolicity;
        let N_op = 1./(-1. + (valley.optical_phonon_energy / (BOLTZMANN * self.sc.temperature)).exp());

        let N_op_eff = if ty == PhononType::Emission { N_op + 1. } else { N_op };
        let E = E.unwrap_or_else(|| self.energy());
        let E_ = if ty == PhononType::Emission { E - valley.optical_phonon_energy } else { E + valley.optical_phonon_energy };
        // can't scatter into if we have too low energy
        if E_ <= 0. {
            return 0.;
        }
        // If E is too low we will subsequently divide by zero
        // However, the graph converges as E -> 0⁺
        // So we cap it at a low value
        let E = E.max(1e-6 * EV_TO_J);
        let γE = E * (1. + α * E);
        let γE_ = E_ * (1. + α * E_);

        let A = (2. * (1. + α * E) * (1. + α * E_) + α * (γE + γE_)).powi(2);
        let B = -2. * α * γE.sqrt() * γE_.sqrt() * (4. * (1. + α * E) * (1. + α * E_) + α * (γE + γE_));
        let C = 4. * (1. + α * E) * (1. + α * E_) * (1. + 2. * α * E) * (1. + 2. * α * E_);
        let F0 = (A * ((γE.sqrt() + γE_.sqrt()) / (γE.sqrt() - γE_.sqrt())).abs().ln() + B) / C;

        N_op_eff * ELECTRON_CHARGE.powi(2) * valley.effective_mass().sqrt() * valley.optical_phonon_energy
            / (2f64.sqrt() * PLANCK_BAR_SI.powi(2) * (4. * PI * EPS0))
            * (1. / self.sc.relative_dielectric_hf - 1. / self.sc.relative_dielectric_static)
            * (1. + 2. * α * E_) / γE.sqrt()
            * F0
    }

    // Isotropic
    // E' = E ± E_phonon - E_fi
    // Matches [monte-carlo-gaas-1980]
    pub fn rate_inter_opt_phonon(&self, ty: PhononType, destination_valley_idx: usize, E: Option<f64>) -> f64 {
        let this_valley = self.valley();
        let dest_valley = &self.sc.valleys[destination_valley_idx];

        let phonon_energy = this_valley.intervalley_phonon_energy[destination_valley_idx];
        let E = E.unwrap_or_else(|| self.energy());
        let E_fi = dest_valley.energy - this_valley.energy;
        let E_ = if ty == PhononType::Emission { E - phonon_energy - E_fi } else { E + phonon_energy - E_fi };
        if E_ <= 0. {
            return 0.0;
        }

        let n_dest_valleys = dest_valley.k_center.len();
        // can't scatter into ourselves bozo
        let n_dest_valleys = if destination_valley_idx == self.valley_idx { n_dest_valleys - 1 } else { n_dest_valleys };
        if n_dest_valleys == 0 {
            return 0.0;
        }

        let N_op = 1./(-1. + (this_valley.optical_phonon_energy / (BOLTZMANN * self.sc.temperature)).exp());
        let N_op_eff = if ty == PhononType::Emission { N_op + 1. } else { N_op };

        N_op_eff * n_dest_valleys as f64 * dest_valley.effective_mass().powf(1.5) * self.sc.intervalley_deformation_potential.powi(2) * E_.sqrt()
            / (2f64.sqrt() * PI * self.sc.density * PLANCK_BAR_SI.powi(2) * phonon_energy)
    }

    // Indep. of k', E' = E
    // From [monte-carlo-book-1989.pdf], CW approach
    pub fn rate_impurity(&self, E: Option<f64>) -> f64 {
        let impurity_mean_dist = (3. / (4. * PI * self.sc.impurity_density)).powf(1./3.);
        let charge_per_impurity: f64 = 1.0; // assumed. how to investigate?
        let E = E.unwrap_or_else(|| self.energy());

        PI * self.sc.impurity_density * charge_per_impurity.powi(2) * impurity_mean_dist.powi(2) * (2. / self.valley().effective_mass()).sqrt() * E.sqrt()
    }
}

#[derive(Clone, Copy)]
pub struct StepInfo {
    pub applied_field: [f64; 3],
    pub maximum_assumed_energy: f64,
}

pub struct FlightResult {
    /// For how long were we in free flight?
    pub free_flight_time: f64,

    // TODO: Not needed I think ever
    // Useful for plotting free flights
    // equal to dk/dt over flight
    pub k_acceleration: [f64; 3],
}

impl<'sc> Electron<'sc> {
    /// Does one step in the monte carlo process
    pub fn free_flight<R: Rng>(&mut self, info: &StepInfo, rng: &mut R) -> FlightResult {
        // TODO: This should be cached. Not sure exactly where though
        // Scattering should probably be the semiconductor's responsibility
        let mechs = Electron::all_mechanisms::<R>();

        // Free flight
        let Γ = mechs.iter().map(|m| (m.maximum_rate)(&self, info.maximum_assumed_energy)).sum::<f64>();
        // TODO: f64::EPSILON is not the smallest number that we can ln without getting zero. find it
        let t = -1./Γ * rng.random_range(0f64 ..= 1f64).ln();

        let force = [
            -ELECTRON_CHARGE * info.applied_field[0],
            -ELECTRON_CHARGE * info.applied_field[1],
            -ELECTRON_CHARGE * info.applied_field[2],
        ];

        let k_acceleration = [
            force[0] / PLANCK_BAR_SI,
            force[1] / PLANCK_BAR_SI,
            force[2] / PLANCK_BAR_SI,
        ];

        self.k = [
            self.k[0] + t * k_acceleration[0],
            self.k[1] + t * k_acceleration[1],
            self.k[2] + t * k_acceleration[2],
        ];

        // TODO: Calculate group velocity and group acceleration
        let v = self.velocity();
        let a = [
            force[0] / self.valley().effective_mass(),
            force[1] / self.valley().effective_mass(),
            force[2] / self.valley().effective_mass(),
        ];

        self.pos = [
            self.pos[0] + v[0] * t + a[0] * t.powi(2) / 2.,
            self.pos[1] + v[1] * t + a[1] * t.powi(2) / 2.,
            self.pos[2] + v[2] * t + a[2] * t.powi(2) / 2.,
        ];

        FlightResult {
            free_flight_time: t,
            k_acceleration,
        }
    }

    pub fn scatter<R: Rng>(&mut self, info: &StepInfo, rng: &mut R) -> Option<ScatteringMechanism<'sc, R>> {
        let mechs = Electron::all_mechanisms();
        let Γ = mechs.iter().map(|m| (m.maximum_rate)(&self, info.maximum_assumed_energy)).sum::<f64>();

        let mut r = Γ * rng.random_range(0. ..= 1.);
        for mech in mechs {
            let prob = (mech.rate)(&self);
            r -= prob;
            if r < 0. {
                // This event happened!
                *self = (mech.resulting_state)(&self, rng);
                return Some(mech);
            }
        }
        // self scattering happened
        None
    }
}
