#![allow(unstable_name_collisions)]

use std::sync::{
    Arc,
    mpsc::{self, Sender, Receiver},
    atomic::Ordering,
};
use std::time::Instant;

use gambling_simulator::{consts::{self, BOLTZMANN, ELECTRON_CHARGE, EPS0}, semiconductor::StepInfo, units::{self, Unit}};
use gambling_simulator::semiconductor::{Semiconductor, Electron};
use gambling_simulator::{ensure_send, ensure_sync};

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use plotly::Plot;
use rustfft::FftPlanner;
use num_complex::{Complex, ComplexFloat};
use errorfunctions::RealErrorFunctions;

use atomic_float::AtomicF64;

use crate::common::plot_utils::UnitPlotter;
mod common;

#[derive(Clone)]
struct Mesh1D {
    // first cell starts at x_start, last cell ends at x_end
    // cell i contains all space between [x_start + Δx *i, x_start + Δx * (i+1)]
    // Δx = (x_end - x_start)/n_cells
    x_start: f64, x_end: f64,
    /// area of each cell
    /// each cell assumed square
    cross_section_area: f64,
    n_cells: usize,
}

impl Mesh1D {
    fn length(&self) -> f64 {
        self.x_end - self.x_start
    }

    fn delta_x(&self) -> f64 {
        self.length() / self.n_cells as f64
    }

    fn idx_to_pos(&self, idx: usize) -> f64 {
        (idx as f64 / self.n_cells as f64) * (self.x_end - self.x_start) + self.x_start
    }

    /// Returns `(i, j, alpha)` where cell `i` has `1-alpha` strength, `j` has `alpha` strength
    /// If we are out of bounds, return one index corresponding either to the first or last element
    fn pos_to_fractional_idx(&self, x: f64) -> Result<(usize, usize, f64), isize> {
        let idx_float = (x - self.x_start) / self.length() * self.n_cells as f64;
        let i = (idx_float - 0.5).floor();
        let j = (idx_float + 0.5).floor();
        let alpha = (idx_float + 0.5) % 1.;

        if i < -1. {
            return Err(-1);
        }
        if i < 0. {
            return Ok((0, 0, 0.));
        }
        if j >= (self.n_cells+1) as f64 {
            return Err(self.n_cells as isize);
        }
        if j >= self.n_cells as f64 {
            return Ok((self.n_cells-1, self.n_cells-1, 0.));
        }
        let (i, j) = (i as usize, j as usize);
        Ok((i, j, alpha))
    }

    // Like pos_to_fractional_idx For a half-grid, with cells [x_start + Δx * (i+0.5), x_start + Δx * (i+1.5)]
    fn pos_to_fractional_idx_in_half_grid(&self, x: f64) -> Result<(usize, usize, f64), isize> {
        let idx_float = (x - self.x_start) / self.length() * self.n_cells as f64;
        let i = (idx_float - 1.0).floor();
        let j = idx_float.floor();
        let alpha = idx_float % 1.;

        if i < -1. {
            return Err(-1);
        }
        if i < 0. {
            return Ok((0, 0, 0.));
        }
        if j >= self.n_cells as f64 {
            return Err((self.n_cells-1) as isize);
        }
        if j >= (self.n_cells-1) as f64 {
            return Ok((self.n_cells-2, self.n_cells-2, 0.));
        }

        let (i, j) = (i as usize, j as usize);
        Ok((i, j, alpha))
    }

    fn position_inside_cell<R: Rng>(&self, r: &mut R, cell_idx: usize) -> [f64; 3] {
        let cross_section_len = self.cross_section_area.sqrt();
        let y = r.random_range(-cross_section_len/2. .. cross_section_len/2.);
        let z = r.random_range(-cross_section_len/2. .. cross_section_len/2.);
        let start = cell_idx as f64 * self.delta_x() + self.x_start;
        let end = (cell_idx + 1) as f64 * self.delta_x() + self.x_start;

        let x = r.random_range(start .. end);
        [x, y, z]
    }

    fn volume_of_cell(&self) -> f64 {
        self.delta_x() * self.cross_section_area
    }
}


struct PoissonState {
    // Written to by MC threads, read (and cleared) by poisson thread
    // Indices correspond to whole-index points from the mesh
    // Number of superparticles, not physical particles
    cell_n_supercarriers: Vec<AtomicF64>,

    // Written to by poisson thread, read by MC threads
    // Indices correspond to half-grid cells (length = mesh.n_cells-1)
    efield: Vec<AtomicF64>,

    // Cleared by poisson thread, added to by MC thread
    superparticle_flux: (AtomicF64, AtomicF64),

    // How many q_0 does each simulated electron represent?
    superparticle_factor: f64,
    total_superparticles: f64,
}

impl PoissonState {
    fn uninit(n_cells: usize, superparticle_factor: f64, total_superparticles: f64) -> PoissonState {
        PoissonState {
            cell_n_supercarriers: std::iter::repeat_with(|| AtomicF64::new(0.)).take(n_cells).collect(),
            efield: std::iter::repeat_with(|| AtomicF64::new(0.)).take(n_cells-1).collect(),
            superparticle_factor, total_superparticles,
            superparticle_flux: (AtomicF64::new(0.), AtomicF64::new(0.)),
        }
    }
    fn reset(&self) {
        for c in &self.cell_n_supercarriers {
            c.store(0., Ordering::Relaxed);
        }
        for e in &self.efield {
            e.store(0., Ordering::Relaxed);
        }
    }
}

fn _test_gridstate_sync() {
    ensure_sync::<PoissonState>();
}

pub enum ElectronInitializationMethod {
    /// n(x) = N_D(x)
    MatchDonorDensity,
    /// Approximate thermal equilibrium based on [del-alamo-2012, p.162], delta doping
    ApproxThermalEquib,

}

struct Simulator<R: Rng + SeedableRng + Send, S> {
    rng: R,
    mesh: Mesh1D,
    /// One semiconductor instance per cell
    /// semiconductor[i].impurity_density is number of donors in cell #i
    semiconductors: Vec<Arc<Semiconductor>>,
    /// References self.semiconductor
    /// Initialized based on semiconductor donor density
    electrons: Vec<Electron>,
    poisson_state: Arc<PoissonState>,
    time: f64,

    /// How often in time should poisson solving happen?
    poisson_solving_interval: f64,

    /// `(fn, interval) = self.callbacks[i];`, fn being called on current Simulator instance every interval of time
    /// Each callback is initially called at the start of each simulation
    /// Note the simulator's electrons will be empty as the MC threads have ownership of electrons.
    callbacks: Vec<(Box<dyn Fn(&mut Simulator<R, S>)>, f64)>,
    callback_state: S,

    // Boundary condition for electric field: E(0) = E(L) = applied_field
    applied_ex: f64,

    barfactory: MultiProgress,

}


#[derive(Debug)]
enum MessageToMCThread {
    /// Update `poisson_state`
    Accumulate,
    /// Limited-time free flight
    FreeFlight { until_time: f64 },
    /// Return
    Done,
}

#[derive(Debug)]
enum MessageToPoissonThread {
    Accumulated,
    SimulatedUntil(f64),
}

fn _ensure_messages_are_send() {
    ensure_send::<MessageToMCThread>();
    ensure_send::<MessageToPoissonThread>();
}

fn split_rng<R: Rng + SeedableRng>(r: &mut R) -> R {
    R::from_rng(r)
}

const FMT_POISSON_BAR: &'static str = "[{elapsed_precise}/{eta_precise}] {wide_bar:.cyan/blue} {pos:>7}/{len:7} {msg}";
const FMT_THREAD_SPINNER: &'static str = "[{prefix}] {msg}";

fn monte_carlo_simulator<R: Rng>(
    barfactory: MultiProgress,
    thread_idx: usize,
    mut electrons: Vec<Electron>,
    mut rng: R,
    semiconductors: Vec<Arc<Semiconductor>>,
    mesh: Mesh1D,
    tx_poisson: Sender<MessageToPoissonThread>,
    rx_mc: Receiver<MessageToMCThread>,
    poisson_state: Arc<PoissonState>,
) -> Vec<Electron> {
    let mut start_time = 0.;
    let mut remaining_free_flight_durations: Vec<f64> = std::iter::repeat(0.).take(electrons.len()).collect();
    let maximum_assumed_energy = units::MEV::to_si(2000.);

    let mut step_info = StepInfo {
        applied_field: [0., 0., 0.,],
        maximum_assumed_energy,
        scattering_mechanisms: Semiconductor::all_mechanisms::<R>(),
    };

    let spinner = barfactory.add(ProgressBar::new_spinner());
    spinner.set_style(ProgressStyle::with_template(FMT_THREAD_SPINNER).unwrap());
    spinner.set_prefix(format!("MC #{thread_idx:#>2}"));

    let mut total_scattering = 0;
    let wall_time_started = Instant::now();
    let mut wall_time_spent_waiting = 0.;
    let mut waited_since = Instant::now();

    let mut n_superparticles_left_domain_total = (0, 0,); // left side, right side
    let (mut rate_left, mut rate_right) = (0., 0.,);

    // Rates for injecting carriers to keep equilibrium

    while let Ok(msg) = rx_mc.recv() {
        wall_time_spent_waiting += waited_since.elapsed().as_secs_f64();

        // Update spinner
        spinner.inc(1);
        let elapsed = wall_time_started.elapsed().as_secs_f64();
        spinner.set_message(format!(
            "{:>5.3} k e⁻, {:.3} k ff-sc/s, {:>4.1}% time spent waiting. {:.3}/{:.3} e⁻ leaving/ps (expect {:.3}/{:.3})",
            (electrons.len() as f64) / 1e3,
            total_scattering as f64 / elapsed / 1e3,
            wall_time_spent_waiting / elapsed * 100.,
            (n_superparticles_left_domain_total.0 as f64 * poisson_state.superparticle_factor / (electrons.len() as f64 / poisson_state.total_superparticles)) / units::PS::from_si(start_time),
            (n_superparticles_left_domain_total.1 as f64 * poisson_state.superparticle_factor / (electrons.len() as f64 / poisson_state.total_superparticles)) / units::PS::from_si(start_time),
            rate_left / units::PS::from_si(1.),
            rate_right / units::PS::from_si(1.),
        ));
        match msg {
            MessageToMCThread::Accumulate => {
                // todo!("Accumulate");
                for el in &electrons {
                    let (i, j, alpha) = match mesh.pos_to_fractional_idx(el.pos[0]) {
                        Ok(x) => x,
                        Err(_i) => continue, // out of bounds, should be handled by FreeFlight
                    };
                    // TODO: Is Relaxed the correct ordering here?
                    // According to https://en.cppreference.com/w/cpp/atomic/memory_order.html#Relaxed_ordering Relaxed should be fine here
                    poisson_state.cell_n_supercarriers[i].fetch_add(1. - alpha, Ordering::Release);
                    poisson_state.cell_n_supercarriers[j].fetch_add(alpha, Ordering::Release);
                }
                tx_poisson.send(MessageToPoissonThread::Accumulated).expect("Couldn't reply to poisson thread!");
            }
            MessageToMCThread::FreeFlight { until_time }  => {
                let electrons_to_process = electrons.drain(..).collect::<Vec<_>>();
                let sim_time = until_time - start_time;

                // Add electrons on left and right
                let mut n_superparticles_left_domain = (0, 0);

                let mut total_velocity_left_domain = 0.;

                for (mut el, remaining_free_flight_duration) in electrons_to_process.into_iter().zip(remaining_free_flight_durations.iter_mut()) {
                    let mut electron_flight_duration = 0.;
                    let mut left_bounds = false;
                    loop {
                        // Sync electron with the cell it's in and get the electric field
                        match (mesh.pos_to_fractional_idx(el.pos[0]), mesh.pos_to_fractional_idx_in_half_grid(el.pos[0])) {
                            (Ok(sc_grid_idx), Ok(efield_grid_idx)) => {
                                if sc_grid_idx.2  < 0.5 {
                                    el.sc = semiconductors[sc_grid_idx.0].clone();
                                } else {
                                    el.sc = semiconductors[sc_grid_idx.1].clone();
                                }

                                let field_i = poisson_state.efield[efield_grid_idx.0].load(Ordering::Acquire);
                                let field_j = poisson_state.efield[efield_grid_idx.1].load(Ordering::Acquire);
                                let field = field_i * (1. - efield_grid_idx.2) + field_j * efield_grid_idx.2;
                                step_info.applied_field[0] = field;
                            }

                            (Err(i), _) | (_, Err(i)) => {
                                left_bounds = true;
                                total_velocity_left_domain += el.velocity()[0];

                                if i == -1 {
                                    n_superparticles_left_domain.0 += 1;
                                } else {
                                    n_superparticles_left_domain.1 += 1;
                                }
                                break;
                            }
                        }

                        // Will the time this flight ends at be after until_time
                        if start_time + electron_flight_duration + *remaining_free_flight_duration > until_time {
                            // This flight ends after until_time, perform half the flight and break
                            let flight_time = until_time - (start_time + electron_flight_duration);
                            el.free_flight(flight_time, &step_info);
                            *remaining_free_flight_duration -= flight_time;
                            break;
                        }
                        // Otherwise we can complete the whole flight
                        el.free_flight(*remaining_free_flight_duration, &step_info);
                        electron_flight_duration += *remaining_free_flight_duration;

                        // Scatter and start new flight
                        el.scatter(&step_info, &mut rng);
                        total_scattering += 1;
                        *remaining_free_flight_duration = el.free_flight_time(&mut rng, &step_info);
                    }
                    if !left_bounds {
                        electrons.push(el);
                    }
                }

                // Repopulate electrons that left
                // Calculate (estimated) drift and thermal velocity on left and right side

                let n_superparticles_left_domain_together = n_superparticles_left_domain.0 + n_superparticles_left_domain.1;

                let mut n_superparticles_entered_domain = (0, 0);

                if n_superparticles_left_domain_together > 0 {
                    let mean_velocity_left_domain = total_velocity_left_domain / n_superparticles_left_domain_together as f64;

                    let sc_left = &semiconductors[0];
                    let sc_right = semiconductors.last().unwrap();
                    let vth_left = (BOLTZMANN * sc_left.temperature / sc_left.valleys[0].effective_mass()).sqrt();
                    let vth_right = (BOLTZMANN * sc_right.temperature / sc_right.valleys[0].effective_mass()).sqrt();

                    let efield_left = poisson_state.efield[0].load(Ordering::Acquire);
                    let efield_right = poisson_state.efield.last().unwrap().load(Ordering::Acquire);

                    let ve_left = sc_left.approx_drift_velocity([efield_left, 0., 0.,])[0];
                    let y0_left = ve_left / vth_left / (2.).sqrt();
                    rate_left = mesh.cross_section_area * sc_left.impurity_density * vth_left / (2.).sqrt() * (
                        (-y0_left.powi(2)).exp() + y0_left * y0_left.erf() + y0_left
                    );

                    let ve_right = -sc_right.approx_drift_velocity([efield_right, 0., 0.,])[0];
                    let y0_right = ve_right / vth_right / (2.).sqrt();
                    rate_right = mesh.cross_section_area * sc_right.impurity_density * vth_right / (2.).sqrt() * (
                        (-y0_right.powi(2)).exp() + y0_right * y0_right.erf() + y0_right
                    );

                    for _ in 0..n_superparticles_left_domain_together {
                        let mut el = Electron::thermalized(&mut rng, semiconductors[0].clone(), 0, [0., 0., 0.,], [mean_velocity_left_domain, 0., 0.,]);
                        if el.k[0] > 0. {
                            // Moving rightwards, put in the left of the domain
                            let i = 0;
                            let mut pos = mesh.position_inside_cell(&mut rng, i);
                            pos[0] = mesh.x_start + mesh.delta_x()/2.;
                            el.pos = pos;
                            n_superparticles_entered_domain.0 += 1;
                        } else {
                            // Moving leftwards, put in the right of the domain
                            let i = mesh.n_cells-1;
                            let mut pos = mesh.position_inside_cell(&mut rng, i);
                            pos[0] = mesh.x_end - mesh.delta_x()/2.;
                            el.pos = pos;
                            n_superparticles_entered_domain.1 += 1;
                        }
                        electrons.push(el);
                    }
                }

                n_superparticles_left_domain_total.0 += n_superparticles_left_domain.0;
                n_superparticles_left_domain_total.1 += n_superparticles_left_domain.1;
                poisson_state.superparticle_flux.0.fetch_add((n_superparticles_left_domain.0 as f64 - n_superparticles_entered_domain.0 as f64) / (sim_time * mesh.cross_section_area), Ordering::AcqRel);
                poisson_state.superparticle_flux.1.fetch_add((n_superparticles_left_domain.1 as f64 - n_superparticles_entered_domain.1 as f64) / (sim_time * mesh.cross_section_area), Ordering::AcqRel);

                start_time = until_time;
                tx_poisson.send(MessageToPoissonThread::SimulatedUntil(until_time)).expect("Couldn't reply to poisson thread!");
            }
            MessageToMCThread::Done => {
                return electrons;
            }
        }
        waited_since = Instant::now();
    }
    spinner.abandon_with_message("died :(");
    panic!("Failed to receive a message from poisson thread!")
}

impl<R: Rng + SeedableRng + Send, S> Simulator<R, S> {
    fn new(
        mut rng: R,
        mesh: Mesh1D,
        n_superparticles: usize,
        semiconductors: Vec<Arc<Semiconductor>>,
        initialization_method: ElectronInitializationMethod,
        poisson_solving_interval: f64,
        callback_state: S,
    ) -> Self {
        let total_physical_electrons: f64 = semiconductors.iter().map(|x| x.impurity_density * mesh.volume_of_cell()).sum();
        let superparticle_factor = total_physical_electrons / n_superparticles as f64;

        let poisson_state = Arc::new(PoissonState::uninit(mesh.n_cells, superparticle_factor, n_superparticles as f64));
        assert_eq!(mesh.n_cells, semiconductors.len());

        let electrons: Vec<Electron> = match initialization_method {
            ElectronInitializationMethod::MatchDonorDensity => {
                (0..n_superparticles).map(|i| {
                    let mut n = i as f64 / n_superparticles as f64 * total_physical_electrons;
                    for (i, sc) in semiconductors.iter().enumerate() {
                        n -= sc.impurity_density * mesh.volume_of_cell();
                        if n < 0. {
                            let pos = mesh.position_inside_cell(&mut rng, i);
                            return Electron::thermalized(&mut rng, sc.clone(), 0, pos, [0., 0., 0.]);
                        }
                    }
                    unreachable!()
                })
                .collect::<Vec<_>>()
            }
            ElectronInitializationMethod::ApproxThermalEquib => unimplemented!("Approximating thermal equilibrium has not been implemented yet!"),
        };
        // Shuffle electrons for extra performance boost (:
        let electrons = {
            let mut electrons = electrons;
            electrons.shuffle(&mut rng);
            electrons
        };

        let barfactory = MultiProgress::new();
        barfactory.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(5));

        Simulator {
            rng,
            mesh,
            semiconductors,
            electrons,
            poisson_state,
            poisson_solving_interval,
            time: 0.,
            applied_ex: 0.,
            callbacks: Vec::new(),
            barfactory,
            callback_state,
        }
    }

    fn register_callback(&mut self, callback: Box<dyn Fn(&mut Self)>, interval: f64) {
        self.callbacks.push((callback, interval));
    }

    fn simulate(&mut self, for_time: f64) {
        let poisson_progress_bar = self.barfactory.add(ProgressBar::new((for_time / self.poisson_solving_interval) as u64));
        poisson_progress_bar.set_style(ProgressStyle::with_template(FMT_POISSON_BAR).unwrap());
        poisson_progress_bar.set_prefix("Poisson");
        let poisson_info = self.barfactory.add(ProgressBar::new_spinner());
        poisson_info.set_style(ProgressStyle::with_template(FMT_THREAD_SPINNER).unwrap());
        poisson_info.set_prefix(format!("Poisson"));

        let mut callbacks_called_at: Vec<f64> = std::iter::repeat(f64::NEG_INFINITY).take(self.callbacks.len()).collect();
        let wall_time_started = Instant::now();
        let mut wall_time_spent_synchronizing = 0.;

        // This main thread is the poisson solving thread. We start off by creating a number of threads for simulating each particle
        let n_threads = 32;//num_cpus::get();

        // Send each electron to one thread
        let thread_electrons = {
            let mut thread_electrons: Vec<Vec<Electron>> = std::iter::repeat_with(|| Vec::new()).take(n_threads).collect();
            for (i, e) in self.electrons.drain(..).enumerate() {
                thread_electrons[i % n_threads].push(e);
            }
            thread_electrons
        };

        std::thread::scope(|sc| {
            let (thread_handles, (txs_mc, rxs_poisson)): (Vec<_>, (Vec<_>, Vec<_>)) =
                thread_electrons.into_iter().enumerate().map(|(thread_idx, electrons)| {
                    let (tx_mc, rx_mc) = mpsc::channel::<MessageToMCThread>();
                    let (tx_poisson, rx_poisson) = mpsc::channel::<MessageToPoissonThread>();
                    let poisson_state = self.poisson_state.clone();
                    let semiconductors = self.semiconductors.clone();
                    let rng = split_rng(&mut self.rng);
                    let barfactory = self.barfactory.clone();
                    let mesh = self.mesh.clone();

                    let handle = sc.spawn(move || monte_carlo_simulator(
                        barfactory, thread_idx, electrons, rng, semiconductors, mesh, tx_poisson, rx_mc, poisson_state
                    ));
                    (handle, (tx_mc, rx_poisson))
                }).unzip();

            let mut t = 0.;
            loop {
                poisson_progress_bar.inc(1);
                let elapsed = wall_time_started.elapsed().as_secs_f64();
                poisson_progress_bar.set_message(format!("t = {}", units::PS::format(t)));
                poisson_info.set_message(format!(
                    "{:.3} ps/s ({:.1} dBs/s). {:>4.3}% time spent synchronizing",
                    units::PS::from_si(t) / elapsed,
                    (t/elapsed).log10()*10.,
                    wall_time_spent_synchronizing / elapsed * 100.,
                ));

                let delta_t = if t + self.poisson_solving_interval < for_time {
                    self.poisson_solving_interval
                } else {
                    // TODO: Should simulate for the last interval?
                    break;
                };

                let sync_start = Instant::now();
                // Clear previous iteration's charge
                for i in 0..self.mesh.n_cells {
                    self.poisson_state.cell_n_supercarriers[i].store(0., Ordering::Release);
                }
                self.poisson_state.superparticle_flux.0.store(0., Ordering::Release);
                self.poisson_state.superparticle_flux.1.store(0., Ordering::Release);

                // Collect charge from all threads
                self.poisson_state.reset();
                for tx in &txs_mc {
                    tx.send(MessageToMCThread::Accumulate).expect("Could not send Accumulate to MC thread");
                }
                // Synchronize, blocking
                for rx in &rxs_poisson {
                    match rx.recv() {
                        Ok(MessageToPoissonThread::Accumulated) => {},
                        Ok(m) => panic!("MC thread out of sync: got {m:?} while expecting Accumulated"),
                        Err(_) => panic!("MC thread died"),
                    }
                }

                // Solve poisson. Boundary condition: efield[-1] = self.applied_field
                let mut acc = self.applied_ex;
                for i in 0..self.mesh.n_cells - 1 {
                    let n_supercarriers_in_cell = self.poisson_state.cell_n_supercarriers[i].load(Ordering::Acquire);
                    let n_physical_carriers_in_cell = n_supercarriers_in_cell * self.poisson_state.superparticle_factor;
                    let n_donors = self.semiconductors[i].impurity_density * self.mesh.volume_of_cell();

                    let rho = ELECTRON_CHARGE * (n_donors - n_physical_carriers_in_cell);
                    let rho_v = rho / self.mesh.volume_of_cell();
                    acc += rho_v / (self.semiconductors[i].relative_dielectric_static * EPS0) * self.mesh.delta_x();
                    self.poisson_state.efield[i].store(acc, Ordering::Release);
                }

                // Let all Monte Carlo threads simulate
                for tx in &txs_mc {
                    tx.send(MessageToMCThread::FreeFlight { until_time: t + delta_t } ).expect("Could not send FreeFlight to MC thread");
                }
                t += delta_t;
                self.time += delta_t;
                wall_time_spent_synchronizing += sync_start.elapsed().as_secs_f64();

                // Synchronize, blocking
                for rx in &rxs_poisson {
                    match rx.recv() {
                        Ok(MessageToPoissonThread::SimulatedUntil(until_time)) => {
                            assert!(
                                (until_time - t).abs() < units::PS::to_si(1e-6),
                                "Poisson thread is out of sync! Time should be {:.3}, got {:.3}(delta of {:.7})",
                                units::PS::format(until_time), units::PS::format(t), units::PS::format(until_time),
                            );
                        },
                        Ok(m) => panic!("MC thread out of sync: got {m:?} while expecting SimulatedUntil(...)"),
                        Err(_) => panic!("MC thread died"),
                    }
                }

                // Check which callbacksh should be called
                // We need to move callbacks out of self because each callback take a mutable reference to self
                // So we can't call a callback while iterating over self

                let callbacks: Vec<_> = self.callbacks.drain(..).collect();

                for ((cb, interval), last_called) in callbacks.iter().zip(callbacks_called_at.iter_mut()) {
                    if t > *last_called + interval {
                        *last_called = t;
                        cb(self);
                    }
                }
                self.callbacks.extend(callbacks);
            }
            // We are done, Stop all threads and return all electrons
            for tx in &txs_mc {
                tx.send(MessageToMCThread::Done).expect("Could not send Done to MC thread");
            }
            for handle in thread_handles {
                match handle.join() {
                    Ok(electrons) => self.electrons.extend(electrons),
                    Err(_) => panic!("Thread didn't die after Done was sent"),
                }
            }

        });
    }
}

const N_SUPERPARTICLES: usize = 10000;

fn main() {
    let mesh = Mesh1D {
        x_start: units::NM::to_si(-500.),
        x_end: units::NM::to_si(500.),
        cross_section_area: units::UM2::to_si(0.07),
        n_cells: 1000,
    };

    let doping_density_1 = units::PER_CM_CUBED::to_si(6.0e15);
    let doping_density_2 = units::PER_CM_CUBED::to_si(6.0e15);

    let doping_count_1 = doping_density_1 * mesh.volume_of_cell();
    let doping_count_2 = doping_density_2 * mesh.volume_of_cell();
    let sim_time = units::PS::to_si(50.);

    eprintln!("{doping_count_1:.3} e⁻ per cell in region 1, {:.3} e⁻ total", doping_density_1 * -mesh.x_start * mesh.cross_section_area);
    eprintln!("{doping_count_2:.3} e⁻ per cell in region 2, {:.3} e⁻ total", doping_density_2 * mesh.x_end * mesh.cross_section_area);

    let semiconductors = (0..mesh.n_cells).map(|i| {
        let x = mesh.idx_to_pos(i);

        let mut sc = Semiconductor::GaAs(300.);
        sc.impurity_density = if x < 0. { doping_density_1 } else { doping_density_2 };
        Arc::new(sc)
    }).collect();

    let poisson_solving_interval = units::PS::to_si(0.005);

    struct Plots {
        carrier_density: (Plot, UnitPlotter<units::NM, units::ELECTRONS_PER_CM_CUBED>),
        efield: (Plot, UnitPlotter<units::NM, units::KV_PER_CM>),
        voltage: (Plot, UnitPlotter<units::NM, units::VOLT>),

        voltages_over_time: Vec<(f64, f64)>,
        current_density_over_time: Vec<(f64, f64)>,
    }
    let plots = {
        let carrier_density = UnitPlotter::new("Charge density", "x", r"\rho_v");
        let efield = UnitPlotter::new("E-field", "x", "E_x");
        let voltage = UnitPlotter::new("Voltage", "x", "V");


        Plots {
            carrier_density: (carrier_density.make_plot(), carrier_density),
            efield: (efield.make_plot(), efield),
            voltage: (voltage.make_plot(), voltage),
            voltages_over_time: Vec::new(),
            current_density_over_time: Vec::new(),
        }
    };

    let mut simulator = Simulator::new(
        ChaCha8Rng::from_os_rng(),
        mesh,
        N_SUPERPARTICLES,
        semiconductors,
        ElectronInitializationMethod::MatchDonorDensity,
        poisson_solving_interval,
        plots,
    );

    let applied_voltage = units::VOLT::to_si(100e-3);
    simulator.applied_ex = -applied_voltage / simulator.mesh.length();

    let blah_bar = simulator.barfactory.insert(0, ProgressBar::new_spinner().with_style(ProgressStyle::with_template(FMT_THREAD_SPINNER).unwrap()));
    blah_bar.set_prefix("Plotter");

    simulator.register_callback(
        Box::new(move |sim| {
            let plots = &mut sim.callback_state;
            let t = units::PS::format(sim.time);
            blah_bar.set_message(format!("Adding at at t = {t}"));

            let x = (0..sim.mesh.n_cells)
                .map(|x| sim.mesh.idx_to_pos(x));

            let superparticle_conc = sim.poisson_state.cell_n_supercarriers.iter().map(|x| x.load(Ordering::Acquire) / sim.mesh.volume_of_cell());
            let conc_relative = superparticle_conc
                .zip(&sim.semiconductors)
                .map(|(s_conc, sc)| ELECTRON_CHARGE * (s_conc * sim.poisson_state.superparticle_factor - sc.impurity_density));

            let trace = plots.carrier_density.1.make_trace(x, conc_relative)
                .name(format!("t = {t}"));
            plots.carrier_density.0.add_trace(trace);

            let x = (0..sim.mesh.n_cells-1)
                .map(|x| sim.mesh.idx_to_pos(x) + sim.mesh.delta_x()/2.);
            let efield = sim.poisson_state.efield.iter().map(|x| x.load(Ordering::Acquire));

            let trace = plots.efield.1.make_trace(x, efield)
                .name(format!("t = {t}"));
            plots.efield.0.add_trace(trace);


            // Integrate to get voltage
            let mut v = std::iter::repeat(0.).take(sim.mesh.n_cells-2).collect::<Vec<_>>();
            let mut acc = 0.;
            for i in 0..sim.mesh.n_cells-2 {
                acc += -sim.poisson_state.efield[i].load(Ordering::Acquire) * sim.mesh.delta_x();
                v[i] = acc;
            }

            let x = (0..sim.mesh.n_cells-2)
                .map(|x| sim.mesh.idx_to_pos(x) + sim.mesh.delta_x());

            let trace = plots.voltage.1.make_trace(x, v)
                .name(format!("t = {t}"));
            plots.voltage.0.add_trace(trace);
        }),
        units::PS::to_si(5.0),
    );

    simulator.register_callback(
        Box::new(move |sim| {
            let plots = &mut sim.callback_state;

            // Integrate to get voltage
            let mut v = std::iter::repeat(0.).take(sim.mesh.n_cells-2).collect::<Vec<_>>();
            let mut acc = 0.;
            for i in 0..sim.mesh.n_cells-2 {
                acc += -sim.poisson_state.efield[i].load(Ordering::Acquire) * sim.mesh.delta_x();
                v[i] = acc;
            }

            let junction_voltage = v.last().unwrap() - v.first().unwrap();
            plots.voltages_over_time.push((sim.time, junction_voltage));

            let e_flux_right = sim.poisson_state.superparticle_flux.1.load(Ordering::Acquire) * sim.poisson_state.superparticle_factor;
            // let e_flux_left  = sim.poisson_state.superparticle_flux.0.load(Ordering::Acquire) * sim.poisson_state.superparticle_factor;
            let charge_flux = e_flux_right * ELECTRON_CHARGE;
            plots.current_density_over_time.push((sim.time, charge_flux));
        }),
        units::PS::to_si(0.),
    );

    simulator.simulate(sim_time);

    let plots = simulator.callback_state;

    let plotter_voltage_over_time = UnitPlotter::<units::PS, units::VOLT>::new("Voltage over time", "t", "V");
    let mut plot_voltage_over_time = plotter_voltage_over_time.make_plot();
    let trace = plotter_voltage_over_time.make_trace(
        plots.voltages_over_time.iter().map(|&(t, _v)| t),
        plots.voltages_over_time.iter().map(|&(_t, v)| v),
    );
    plot_voltage_over_time.add_trace(trace);

    let plotter_current_over_time = UnitPlotter::<units::PS, units::A_PER_CM2>::new("Current density over time", "t", "J");
    let mut plot_current_over_time = plotter_current_over_time.make_plot();
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _v)| t),
        plots.current_density_over_time.iter().map(|&(_t, v)| v),
    );
    plot_current_over_time.add_trace(trace);

    let voltage_freq = common::plot_utils::UnitPlotter::<units::THZ, units::DBV2PerHz>::new("Power spectral density", "f", "PSD");
    let mut plot_voltage_freq = voltage_freq.make_plot();

    // Take FFT of voltage
    let n_points = plots.voltages_over_time.len();
    let t = plots.voltages_over_time.last().unwrap().0 - plots.voltages_over_time.first().unwrap().0;
    let df = 1. / t;
    let fs = (0..n_points).map(|i| i as f64 * df).collect::<Vec<_>>();
    let transformer = FftPlanner::<f64>::new().plan_fft_forward(n_points);
    let fft = {
        let mut buffer = plots.voltages_over_time.iter()
            .map(|&(_t, v)| Complex { re: v, im: 0. })
            .collect::<Vec<_>>();
        transformer.process(&mut buffer);
        buffer.into_iter()
            .take(n_points/2)
            .map(|x| x / n_points as f64)
            .map(|x| x.powi(2).abs()) // PSD
            .map(|x| x / df) // Go from PSD per df to PSD per Hz
            .collect::<Vec<_>>()
    };

    let scatter = voltage_freq.make_trace(fs, fft);
    plot_voltage_freq.add_trace(scatter);

    common::write_plots("poisson", "poisson2", [plots.carrier_density.0, plots.efield.0, plots.voltage.0, plot_voltage_over_time, plot_voltage_freq, plot_current_over_time]);

    // Calculate expected values for parameters
    let v_exp = applied_voltage;

    let mobility_linear = 0.85; // m/s / (V/m)
    let r_exp = 1. / (doping_density_1 * ELECTRON_CHARGE * mobility_linear) * simulator.mesh.length() / simulator.mesh.cross_section_area;
    let i_exp = v_exp / r_exp;

    // Integrate johnson noise from f = 0 to f = (1/2pi) kB T / hbar;
    let temp = simulator.semiconductors[0].temperature;
    let f_stop = 1./(2. * std::f64::consts::PI) * consts::BOLTZMANN * temp / consts::PLANCK_BAR_SI;
    let johnson_voltage_noise_density = 4. * consts::BOLTZMANN * temp * r_exp;
    let v_std_exp = (johnson_voltage_noise_density * f_stop).sqrt();

    let johnson_current_noise_density = 4. * consts::BOLTZMANN * temp * 1. / r_exp;
    let i_std_exp = (johnson_current_noise_density * f_stop).sqrt();


    // Voltage stats
    let v_mean = plots.voltages_over_time.iter().map(|(_t, v)| v).sum::<f64>() / plots.voltages_over_time.len() as f64;
    let v_var = (plots.voltages_over_time.iter().map(|(_t, v)| (v - v_mean).powi(2)).sum::<f64>()) / plots.voltages_over_time.len() as f64;
    let v_std = v_var.sqrt();
    simulator.barfactory.println(format!("Voltage: mean = {}, std = {}", units::MILLIVOLT::format(v_mean), units::MILLIVOLT::format(v_std))).unwrap();
    simulator.barfactory.println(format!("    exp: mean = {}, std = {}", units::MILLIVOLT::format(v_exp), units::MILLIVOLT::format(v_std_exp))).unwrap();

    // Current stats
    let j_mean = plots.current_density_over_time.iter().map(|(_t, v)| v).sum::<f64>() / plots.current_density_over_time.len() as f64;
    let j_var = (plots.current_density_over_time.iter().map(|(_t, v)| (v - v_mean).powi(2)).sum::<f64>()) / plots.current_density_over_time.len() as f64;
    let j_std = j_var.sqrt();
    let i_mean = j_mean * simulator.mesh.cross_section_area;
    let i_std = j_std * simulator.mesh.cross_section_area;

    simulator.barfactory.println(format!("Current: mean = {}, std = {}", units::MICROAMP::format(i_mean), units::MICROAMP::format(i_std))).unwrap();
    simulator.barfactory.println(format!("    exp: mean = {}, std = {}", units::MICROAMP::format(i_exp), units::MICROAMP::format(i_std_exp))).unwrap();

    // Calculate resistance
    let r = v_mean / i_mean;
    simulator.barfactory.println(format!("Resistance = {}", units::OHM::format(r))).unwrap();
    simulator.barfactory.println(format!("       Exp = {}", units::OHM::format(r_exp))).unwrap();

}
