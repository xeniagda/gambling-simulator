#![allow(unstable_name_collisions)]

use std::{f64::consts::PI, sync::{
    Arc, atomic::Ordering, mpsc::{self, Receiver, Sender}
}};
use std::time::Instant;

use gambling_simulator::{consts::{BOLTZMANN, ELECTRON_CHARGE, ELECTRON_MASS, EPS0, PLANCK_BAR_SI}, semiconductor::StepInfo, units::{self, Unit}};
use gambling_simulator::semiconductor::{Semiconductor, Electron};
use gambling_simulator::{ensure_send, ensure_sync};

use npyz::WriterBuilder;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use plotly::{Plot, common::{DashType, Line}};
use rand_distr::Poisson;

use atomic_float::AtomicF64;
use clap::{Parser, Subcommand, Args};

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
    // Written to by MC threads, read by poisson thread
    // Cleared by poisson thread before each MC step
    // Indices correspond to whole-index points from the mesh
    // Number of superparticles, not physical particles
    cell_n_supercarriers: Vec<AtomicF64>,

    // Written to by MC thread whenever an electron leaves or enters the semiconductor metal junction
    // Also written to by poisson thread to reestablish voltage across diode
    n_supercarriers_at_metal_contact: (AtomicF64, AtomicF64),

    // metal-contact barrier height (in volts)
    barrier_height: f64,

    // Written to by poisson thread, read by MC threads
    // Indices correspond to half-grid cells (length = mesh.n_cells-1)
    efield: Vec<AtomicF64>,

    // Written to by poisson thread, not used by MC threads
    // Indices correspond to whole-grid cells (length = mesh.n_cells)
    voltage: Vec<AtomicF64>,


    // Cleared by poisson thread, added to by MC thread
    // Represents flux OUT of area, number per second
    superparticle_flux_into_left: AtomicF64,
    superparticle_flux_out_of_left: AtomicF64,
    superparticle_flux_into_right: AtomicF64,
    superparticle_flux_out_of_right: AtomicF64,

    // How many q_0 does each simulated electron represent?
    n_threads: usize,
    superparticle_factor: f64,
    total_superparticles: f64,
}

impl PoissonState {
    fn uninit(barrier_height: f64, n_cells: usize, n_threads: usize, superparticle_factor: f64, total_superparticles: f64) -> PoissonState {
        PoissonState {
            cell_n_supercarriers: std::iter::repeat_with(|| AtomicF64::new(0.)).take(n_cells).collect(),
            n_supercarriers_at_metal_contact: (AtomicF64::new(0.), AtomicF64::new(0.)),
            barrier_height,
            efield: std::iter::repeat_with(|| AtomicF64::new(0.)).take(n_cells-1).collect(),
            voltage: std::iter::repeat_with(|| AtomicF64::new(0.)).take(n_cells).collect(),
            superparticle_factor, total_superparticles,
            superparticle_flux_into_left: AtomicF64::new(0.),
            superparticle_flux_out_of_left: AtomicF64::new(0.),
            superparticle_flux_into_right: AtomicF64::new(0.),
            superparticle_flux_out_of_right: AtomicF64::new(0.),
            n_threads,
        }
    }
    fn reset(&self) {
        for c in &self.cell_n_supercarriers {
            c.store(0., Ordering::SeqCst);
        }
        for e in &self.efield {
            e.store(0., Ordering::SeqCst);
        }
        for v in &self.voltage {
            v.store(0., Ordering::SeqCst);
        }

        self.superparticle_flux_into_left.store(0., Ordering::SeqCst);
        self.superparticle_flux_out_of_left.store(0., Ordering::SeqCst);
        self.superparticle_flux_into_right.store(0., Ordering::SeqCst);
        self.superparticle_flux_out_of_right.store(0., Ordering::SeqCst);
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
    callbacks: Vec<(Box<dyn FnMut(&mut Simulator<R, S>)>, f64)>,
    callback_state: S,

    // Boundary condition for electric field: E(0) = E(L) = applied_field
    applied_voltage: f64,

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

const FMT_POISSON_BAR: &'static str = "[{prefix} {elapsed_precise}/{eta_precise}] {wide_bar:.cyan/blue} {pos:>7}/{len:7} {msg}";
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
    let n_cells = mesh.n_cells;
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

    let mut mean_velocity_per_cell: Vec<_> = std::iter::repeat(0.).take(mesh.n_cells).collect();

    // del Alamo, eq. 7.33
    let richardsson_constant_gaas = 4. * PI * ELECTRON_CHARGE * BOLTZMANN.powi(2) * ELECTRON_MASS / (PLANCK_BAR_SI * 2. * PI).powi(3) * 0.066f64.powi(3) / 0.070;
    let temp = semiconductors[0].temperature;
    let injection_current_from_metal = richardsson_constant_gaas * temp.powi(2) * (-poisson_state.barrier_height / (BOLTZMANN * temp / ELECTRON_CHARGE)).exp();

    while let Ok(msg) = rx_mc.recv() {
        wall_time_spent_waiting += waited_since.elapsed().as_secs_f64();

        // Update spinner
        spinner.inc(1);
        let elapsed = wall_time_started.elapsed().as_secs_f64();
        let mean_velocity = mean_velocity_per_cell.iter().sum::<f64>() / mesh.n_cells as f64;

        spinner.set_message(format!(
            "{:>6.3} k e⁻ᵥ, {:.3} k ff-sc/s, {:>4.1}% time spent waiting. <v> = {}",
            (electrons.len() as f64) / 1e3,
            total_scattering as f64 / elapsed / 1e3,
            wall_time_spent_waiting / elapsed * 100.,
            units::MILLION_CM_PER_SECOND::format(mean_velocity),
        ));
        match msg {
            MessageToMCThread::Accumulate => {
                for el in &electrons {
                    let (i, j, alpha) = match mesh.pos_to_fractional_idx(el.pos[0]) {
                        Ok(x) => x,
                        Err(_i) => panic!("Electron out of bounds after free flight"), // out of bounds, should be handled by FreeFlight
                    };
                    poisson_state.cell_n_supercarriers[i].fetch_add(1. - alpha, Ordering::SeqCst);
                    poisson_state.cell_n_supercarriers[j].fetch_add(alpha, Ordering::SeqCst);
                }
                tx_poisson.send(MessageToPoissonThread::Accumulated).expect("Couldn't reply to poisson thread!");
            }
            MessageToMCThread::FreeFlight { until_time }  => {
                let sim_time = until_time - start_time;
                let mut integrated_velocity_per_cell: Vec<f64> = std::iter::repeat(0.).take(mesh.n_cells).collect();
                let mut integrated_time_per_cell: Vec<f64> = std::iter::repeat(0.).take(mesh.n_cells).collect();

                let count = electrons.len();
                let current_electrons: Vec<_> = std::mem::replace(&mut electrons, Vec::with_capacity(count));
                let current_flight_durations: Vec<_> = std::mem::replace(&mut remaining_free_flight_durations, Vec::with_capacity(count));

                for (mut el, mut remaining_free_flight_duration) in current_electrons.into_iter().zip(current_flight_durations.into_iter()) {
                    let mut electron_got_absorbed = false;
                    let mut electron_finished = false;

                    let mut electron_flight_duration = 0.;
                    loop {
                        // Sync electron with the cell it's in and get the electric field
                        let cell_idx = match (mesh.pos_to_fractional_idx(el.pos[0]), mesh.pos_to_fractional_idx_in_half_grid(el.pos[0])) {
                            (Ok(sc_grid_idx), Ok(efield_grid_idx)) => {
                                let field_i = poisson_state.efield[efield_grid_idx.0].load(Ordering::SeqCst);
                                let field_j = poisson_state.efield[efield_grid_idx.1].load(Ordering::SeqCst);
                                let field = field_i * (1. - efield_grid_idx.2) + field_j * efield_grid_idx.2;
                                step_info.applied_field[0] = field;

                                if sc_grid_idx.2 < 0.5 {
                                    el.sc = semiconductors[sc_grid_idx.0].clone();
                                    sc_grid_idx.0
                                } else {
                                    el.sc = semiconductors[sc_grid_idx.1].clone();
                                    sc_grid_idx.1
                                }
                            }
                            (Err(-1), _) | (_, Err(-1)) => {
                                // Electron reached schottky contact and gets absorbed
                                electron_got_absorbed = true;

                                poisson_state.superparticle_flux_out_of_left.fetch_add(1. / sim_time, Ordering::SeqCst);
                                poisson_state.n_supercarriers_at_metal_contact.0.fetch_add(1., Ordering::SeqCst);

                                break;
                            }
                            (Err(_), _) | (_, Err(_)) => {
                                // Electron reached ohmic contact, leaves domain

                                electron_got_absorbed = true;
                                // Charge ends up at the metal contact

                                poisson_state.superparticle_flux_out_of_right.fetch_add(1. / sim_time, Ordering::SeqCst);
                                poisson_state.n_supercarriers_at_metal_contact.1.fetch_add(1., Ordering::SeqCst);
                                break;
                            }
                        };
                        if electron_finished {
                            break;
                        }

                        // Will the time this flight ends at be after until_time
                        if start_time + electron_flight_duration + remaining_free_flight_duration > until_time {
                            // This flight ends after until_time, perform half the flight and break
                            let flight_time = until_time - (start_time + electron_flight_duration);

                            integrated_velocity_per_cell[cell_idx] += el.velocity()[0] * flight_time;
                            integrated_time_per_cell[cell_idx] += flight_time;

                            el.free_flight(flight_time, &step_info);
                            remaining_free_flight_duration -= flight_time;

                            electron_finished = true;
                            continue;
                        }
                        integrated_velocity_per_cell[cell_idx] += el.velocity()[0] * remaining_free_flight_duration;
                        integrated_time_per_cell[cell_idx] += remaining_free_flight_duration;

                        // Otherwise we can complete the whole flight
                        el.free_flight(remaining_free_flight_duration, &step_info);
                        electron_flight_duration += remaining_free_flight_duration;

                        // Scatter and start new flight
                        el.scatter(&step_info, &mut rng);
                        total_scattering += 1;
                        remaining_free_flight_duration = el.free_flight_time(&mut rng, &step_info);
                    }

                    if !electron_got_absorbed {
                        electrons.push(el);
                        remaining_free_flight_durations.push(remaining_free_flight_duration);
                    }
                }

                // Handle ohmic contact:
                // The divergence of the electric field gives the charge concentration. If we have too few electrons, inject a few
                const N_CELLS_AVERAGED: usize = 100;
                let div_e = (
                        poisson_state.efield[n_cells-2].load(Ordering::SeqCst) - poisson_state.efield[n_cells-2 - N_CELLS_AVERAGED].load(Ordering::SeqCst)
                    ) / (mesh.delta_x() * N_CELLS_AVERAGED as f64);
                let rho_v = div_e * (EPS0 * semiconductors.last().unwrap().relative_dielectric_static);
                let rho = rho_v * (mesh.volume_of_cell() * N_CELLS_AVERAGED as f64);
                if rho > 0. {
                    // Deficiency of electrons, inject a few (:
                    let n_superparticles_should_be_injected = rho / ELECTRON_CHARGE / poisson_state.superparticle_factor;
                    let n_superparticles_in_this_thread = n_superparticles_should_be_injected / poisson_state.n_threads as f64;
                    let dist = Poisson::new(n_superparticles_in_this_thread).unwrap();
                    let n_to_inject = rng.sample(dist) as u64;

                    let sc_right = semiconductors.last().unwrap();
                    for _ in 0..n_to_inject {
                        let mut el = Electron::thermalized(&mut rng, sc_right.clone(), 0, [0., 0., 0.,], [0., 0., 0.,]);
                        el.k[0] = -el.k[0].abs();

                        let inj_cell_idx = mesh.n_cells-1;
                        let mut pos = mesh.position_inside_cell(&mut rng, inj_cell_idx);
                        let n = rng.random_range(0. .. N_CELLS_AVERAGED as f64);
                        pos[0] = mesh.x_end - (n + 0.5) * mesh.delta_x()/2.;
                        el.pos = pos;

                        let free_flight_time = el.free_flight_time(&mut rng, &step_info);
                        electrons.push(el);
                        remaining_free_flight_durations.push(free_flight_time);
                    }
                    // bookkeep the injected particles
                    poisson_state.superparticle_flux_into_right.fetch_add(n_to_inject as f64 / sim_time, Ordering::SeqCst);
                    poisson_state.n_supercarriers_at_metal_contact.0.fetch_sub(n_to_inject as f64, Ordering::SeqCst);
                }

                // Handle Schottky contact
                let n_electrons_should_be_injected = injection_current_from_metal * sim_time * mesh.cross_section_area / ELECTRON_CHARGE;
                let n_superparticles_should_be_injected = n_electrons_should_be_injected / poisson_state.superparticle_factor;
                let n_superparticles_in_this_thread = n_superparticles_should_be_injected * (electrons.len() as f64 / poisson_state.total_superparticles);
                let dist = Poisson::new(n_superparticles_in_this_thread).unwrap();
                let n_to_inject = rng.sample(dist) as u64;

                let sc_left = semiconductors[0].clone();
                for _ in 0..n_to_inject {
                    let mut el = Electron::thermalized(&mut rng, sc_left.clone(), 0, [0., 0., 0.,], [0., 0., 0.,]);
                    el.k[0] = el.k[0].abs();

                    let inj_cell_idx = 0;
                    let mut pos = mesh.position_inside_cell(&mut rng, inj_cell_idx);
                    pos[0] = mesh.x_start + mesh.delta_x() / 2.;
                    el.pos = pos;

                    let free_flight_time = el.free_flight_time(&mut rng, &step_info);
                    electrons.push(el);
                    remaining_free_flight_durations.push(free_flight_time);
                }
                // bookkeep the injected particles
                poisson_state.superparticle_flux_into_left.fetch_add(n_to_inject as f64 / sim_time, Ordering::SeqCst);
                poisson_state.n_supercarriers_at_metal_contact.1.fetch_sub(n_to_inject as f64, Ordering::SeqCst);

                start_time = until_time;
                tx_poisson.send(MessageToPoissonThread::SimulatedUntil(until_time)).expect("Couldn't reply to poisson thread!");

                mean_velocity_per_cell = integrated_velocity_per_cell.into_iter().zip(integrated_time_per_cell)
                    .map(|(intvel, t)| {
                        if t == 0. {
                            0.
                        } else {
                            intvel / t
                        }
                    })
                    .collect();
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
        barrier_height: f64,
    ) -> Self {
        let total_physical_electrons: f64 = semiconductors.iter().map(|x| x.impurity_density * mesh.volume_of_cell()).sum();
        let superparticle_factor = total_physical_electrons / n_superparticles as f64;

        let poisson_state = Arc::new(PoissonState::uninit(barrier_height, mesh.n_cells, 0, superparticle_factor, n_superparticles as f64));
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
            applied_voltage: 0.,
            callbacks: Vec::new(),
            barfactory,
            callback_state,
        }
    }

    fn register_callback(&mut self, callback: Box<dyn FnMut(&mut Self)>, interval: f64) {
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
        let n_threads = 48;//num_cpus::get();
        Arc::get_mut(&mut self.poisson_state).unwrap().n_threads = n_threads;

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

                // Solve Poisson
                // V[i] represents voltage in region [i Δx, (i+1) Δx]
                // E[i] represent E-field in [(i+1/2) Δx, (i+3/2) Δx]
                // V[-1], V[N] is inside the metal
                // E[-1], E[N-1] is inside the metal
                // Gauss for charge:
                //   (E[i] - E[i-1])/Δx = ρ_v,i / ε
                // Voltage:
                //   (V[i]-V[i-1])/Δx = -E[i-1]
                //
                // BCs:
                //  E[-1] = ρ_s,0 / epsilon (surface charge on metal)
                //  E[N] = -ρ_s,1 / epsilon (surface charge on metal)
                //  V[-1] = 0 (metal grounded)
                //  V[N] = -V_applied

                // We do a two-stage solution. First find a solution E_0, V_0 assuming ρ_s,0 = 0
                // Then we know a solution V with nonzero ρ_s,0 will be V(x) = V_0(x) - x ρ_s,0/ε
                // Therefore we can solve ρ_s,0 = -ε(V[N] - V_0[N])/L
                // And then re-solve E-field

                fn solve_poisson(
                    surface_charge_at_metal_left: f64,
                    poisson_state: &PoissonState,
                    mesh: &Mesh1D,
                    semiconductors: &[Arc<Semiconductor>],
                ) -> f64 {
                    // First stage: Solve E assuming ρ_s,0 = 0
                    let mut e_prev = surface_charge_at_metal_left / (semiconductors[0].relative_dielectric_static * EPS0);
                    for i in 0..mesh.n_cells - 1 {
                        let n_superparticles_in_cell = poisson_state.cell_n_supercarriers[i].load(Ordering::SeqCst);
                        let volume_charge_carriers = n_superparticles_in_cell * poisson_state.superparticle_factor * -ELECTRON_CHARGE / mesh.volume_of_cell();
                        let volume_charge_donors = semiconductors[i].impurity_density * ELECTRON_CHARGE;
                        let rho_v = volume_charge_donors + volume_charge_carriers;

                        let e_here = e_prev + mesh.delta_x() * rho_v / (semiconductors[i].relative_dielectric_static * EPS0);
                        poisson_state.efield[i].store(e_here, Ordering::SeqCst);

                        e_prev = e_here;
                    }
                    // Solve voltage
                    let mut v_prev = 0.;
                    poisson_state.voltage[0].store(v_prev, Ordering::SeqCst);
                    for i in 1..mesh.n_cells {
                        let v_here = v_prev - poisson_state.efield[i-1].load(Ordering::SeqCst) * mesh.delta_x();
                        poisson_state.voltage[i].store(v_here, Ordering::SeqCst);
                        v_prev = v_here;
                    }
                    v_prev
                }

                // First step: Solve voltage and e-field and voltage assuming ρ_s,0 = 0
                let v_0 = solve_poisson(0., &self.poisson_state, &self.mesh, &self.semiconductors);

                // Second step: Solve ρ_s,0 given wanted voltage
                let solved_surface_charge_at_metal_left = -(self.semiconductors[0].relative_dielectric_static * EPS0) * (-self.applied_voltage - v_0) / self.mesh.length();

                let total_superparticles_at_metal = self.poisson_state.n_supercarriers_at_metal_contact.0.load(Ordering::SeqCst)
                    + self.poisson_state.n_supercarriers_at_metal_contact.1.load(Ordering::SeqCst);

                let n_superparticles_at_left_contact = solved_surface_charge_at_metal_left / -ELECTRON_CHARGE / self.poisson_state.superparticle_factor * self.mesh.cross_section_area;
                let n_superparticles_at_right_contact = total_superparticles_at_metal - n_superparticles_at_left_contact;
                self.poisson_state.n_supercarriers_at_metal_contact.0.store(n_superparticles_at_left_contact, Ordering::SeqCst);
                self.poisson_state.n_supercarriers_at_metal_contact.1.store(n_superparticles_at_right_contact, Ordering::SeqCst);

                // Third step: Re-solve voltage and e-field
                solve_poisson(solved_surface_charge_at_metal_left, &self.poisson_state, &self.mesh, &self.semiconductors);

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

                let mut callbacks: Vec<_> = self.callbacks.drain(..).collect();

                for ((cb, interval), last_called) in callbacks.iter_mut().zip(callbacks_called_at.iter_mut()) {
                    if t > *last_called + *interval {
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

fn setup<S>(callback_state: S) -> Simulator<ChaCha8Rng, S> {
    let mesh = Mesh1D {
        // real diode
        // x_start: units::NM::to_si(-50.),
        // x_end: units::NM::to_si(500.),
        // fake diode
        x_start: units::NM::to_si(-2000.),
        x_end: units::NM::to_si(1000.),
        cross_section_area: units::UM2::to_si(0.11),
        n_cells: 5000,
    };

    // real diode
    // let doping_density_1 = units::PER_CM_CUBED::to_si(6e17);
    // let doping_density_2 = units::PER_CM_CUBED::to_si(5e18);

    // fake diode
    let doping_density_1 = units::PER_CM_CUBED::to_si(1e16);
    let doping_density_2 = units::PER_CM_CUBED::to_si(5e17); // fake ohmic contact

    // 200mV barrier sounds about right
    let barrier_height: f64 = units::MILLIVOLT::to_si(200.);

    let poisson_solving_interval = units::PS::to_si(0.005);

    let semiconductors: Vec<_> = (0..mesh.n_cells).map(|i| {
        let x = mesh.idx_to_pos(i);

        let mut sc = Semiconductor::GaAs(300.);
        if x <= 0. {
            sc.impurity_density = doping_density_1;
        } else {
            sc.impurity_density = doping_density_2;
        }

        Arc::new(sc)
    }).collect();

    Simulator::new(
        ChaCha8Rng::from_os_rng(),
        mesh,
        N_SUPERPARTICLES,
        semiconductors,
        ElectronInitializationMethod::MatchDonorDensity,
        poisson_solving_interval,
        callback_state,
        barrier_height,
    )
}

#[derive(Parser)]
struct Command {
    #[command(subcommand)]
    command: Sub,
}

#[derive(Subcommand)]
enum Sub {
    Sweep(DCSweepArgs),
    Single(SingleArgs),
}

#[derive(Args)]
struct DCSweepArgs {
    #[arg(short = 'N', long, default_value_t = 1)]
    n_computers: usize,

    #[arg(short = 'n', long, default_value_t = 0)]
    this_computer: usize,

    #[arg(short = 'o', long)]
    output_folder_name: Option<String>,
}

#[derive(Args)]
struct SingleArgs {
    #[arg(short = 'v', long, allow_negative_numbers(true))]
    voltage: f64,
}

fn main() {
    match Command::parse().command {
        Sub::Sweep(dcsweep_args) => {
            main_dc_sweep(dcsweep_args);
        }
        Sub::Single(single_args) => {
            main_single(single_args);
        }
    }
}

fn main_dc_sweep(args: DCSweepArgs) {
    let sim_time = units::PS::to_si(400.1);
    let sim_time_record_start = units::PS::to_si(40.);

    struct DataPoint {
        at_time: f64,
        applied_voltage: f64,
        measured_voltage: f64,
        current_left: f64,
        current_right: f64,
    }

    let mut simulator = setup(Vec::new());

    simulator.register_callback(Box::new(move |sim| {
            if sim.time < sim_time_record_start {
                return;
            }
            let measured_voltage = sim.poisson_state.voltage.last().unwrap().load(Ordering::SeqCst) - sim.poisson_state.voltage[0].load(Ordering::SeqCst);

            let current_into_left = sim.poisson_state.superparticle_flux_into_left.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor;
            let current_out_of_left = sim.poisson_state.superparticle_flux_out_of_left.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor;
            let current_into_right = sim.poisson_state.superparticle_flux_into_right.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor;
            let current_out_of_right = sim.poisson_state.superparticle_flux_out_of_right.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor;

            let current_left = current_into_left - current_out_of_left;
            let current_right = current_out_of_right - current_into_right;

            sim.callback_state.push(DataPoint {
                at_time: sim.time,
                applied_voltage: sim.applied_voltage,
                measured_voltage: measured_voltage,
                current_left,
                current_right,
            });
        }), units::PS::from_si(0.),
    );

    let n = 40;
    let v_start = units::MILLIVOLT::to_si(-500.);
    let v_stop = units::MILLIVOLT::to_si(300.);
    let all_voltages = (0..(n+1)).map(|i| i as f64 / (n as f64) * (v_stop - v_start) + v_start).collect::<Vec<_>>();

    let my_start = all_voltages.len() * args.this_computer / args.n_computers;
    let next_start = all_voltages.len() * (args.this_computer + 1) / args.n_computers;
    let voltages = &all_voltages[my_start..next_start];

    let voltage_bar = Arc::new(simulator.barfactory.add(ProgressBar::new(voltages.len() as u64)));
    voltage_bar.set_style(ProgressStyle::with_template(FMT_POISSON_BAR).unwrap());
    voltage_bar.set_prefix(format!("Voltage sweep {}≤_<{} / {}", my_start, next_start, all_voltages.len()));

    simulator.register_callback(Box::new({
        let voltage_bar = voltage_bar.clone();
        move |_sim| {
            voltage_bar.tick();
        }
    }), units::PS::from_si(1.));

    for &voltage in voltages {
        voltage_bar.set_message(format!("V = {}", units::MILLIVOLT::format(voltage)));

        simulator.applied_voltage = voltage;
        simulator.time = 0.;
        simulator.simulate(sim_time);

        voltage_bar.inc(1);
    }

    let points = simulator.callback_state;
    simulator.barfactory.println(format!("Got {} data points", points.len())).unwrap();

    // Write the data to a file
    let mut path = std::path::PathBuf::from("./plots/poisson/");
    if let Some(f) = args.output_folder_name {
        path.push(f);
    }
    path.push(format!("voltage_sweep-{}-of-{}.npz", args.this_computer, args.n_computers));

    simulator.barfactory.println(format!("Writing to {}", path.display())).unwrap();

    let outfile = std::fs::File::create(path).expect("Could not open output file");
    let mut writer = npyz::WriteOptions::<f64>::new()
        .default_dtype()
        .shape(&[points.len() as u64, 5])
        .writer(outfile)
        .begin_nd()
        .expect("Could not build npz writer");

    for p in points {
        writer.push(&p.at_time).unwrap();
        writer.push(&p.applied_voltage).unwrap();
        writer.push(&p.measured_voltage).unwrap();
        writer.push(&p.current_left).unwrap();
        writer.push(&p.current_right).unwrap();
    }
    writer.finish().expect("Could not write file");
    simulator.barfactory.println("Written").unwrap();
}

#[allow(unused)]
fn main_single(args: SingleArgs) {
    let sim_time = units::PS::to_si(50.1);

    let applied_voltage = units::VOLT::to_si(args.voltage);

    struct Plots {
        carrier_density: (Plot, UnitPlotter<units::NM, units::ELECTRONS_PER_CM_CUBED>),
        efield: (Plot, UnitPlotter<units::NM, units::KV_PER_CM>),
        voltage: (Plot, UnitPlotter<units::NM, units::VOLT>),

        voltages_over_time: Vec<(f64, f64)>,
        current_density_over_time: Vec<(f64, f64, f64, f64, f64)>, // time, into left side, out of left side, into right side, out of right side

        total_charge_over_time: Vec<(f64, f64, f64, f64)>, // time, volume charge, surface charge left, surface contact right
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
            total_charge_over_time: Vec::new(),
        }
    };

    let mut simulator = setup(plots);
    simulator.applied_voltage = applied_voltage;

    let blah_bar = simulator.barfactory.insert(0, ProgressBar::new_spinner().with_style(ProgressStyle::with_template(FMT_THREAD_SPINNER).unwrap()));
    blah_bar.set_prefix("Plotter");

    simulator.register_callback(
        Box::new(move |sim| {
            let plots = &mut sim.callback_state;
            let t = units::PS::format(sim.time);
            blah_bar.set_message(format!("Adding at at t = {t}"));

            let x = (0..sim.mesh.n_cells)
                .map(|x| sim.mesh.idx_to_pos(x));

            let superparticle_conc = sim.poisson_state.cell_n_supercarriers.iter().map(|x| x.load(Ordering::SeqCst) / sim.mesh.volume_of_cell());
            let conc_relative = superparticle_conc
                .map(|s_conc| ELECTRON_CHARGE * s_conc * sim.poisson_state.superparticle_factor);

            let trace = plots.carrier_density.1.make_trace(x, conc_relative)
                .name(format!("t = {t}"));
            plots.carrier_density.0.add_trace(trace);

            let x = (0..sim.mesh.n_cells-1)
                .map(|x| sim.mesh.idx_to_pos(x) + sim.mesh.delta_x()/2.);
            let efield = sim.poisson_state.efield.iter().map(|x| x.load(Ordering::SeqCst));

            let trace = plots.efield.1.make_trace(x, efield)
                .name(format!("t = {t}"));
            plots.efield.0.add_trace(trace);

            let x = (0..sim.mesh.n_cells)
                .map(|x| sim.mesh.idx_to_pos(x) + sim.mesh.delta_x());

            let v = sim.poisson_state.voltage.iter().map(|v| v.load(Ordering::SeqCst));

            let trace = plots.voltage.1.make_trace(x, v)
                .name(format!("t = {t}"));
            plots.voltage.0.add_trace(trace);
        }),
        units::PS::to_si(10.0),
    );


    // Exponential averaging
    let decay_per_time: f64 = 5. / units::PS::to_si(1.);
    let mut last_current_density_into_left = 0.;
    let mut last_current_density_out_of_left = 0.;
    let mut last_current_density_into_right = 0.;
    let mut last_current_density_out_of_right = 0.;

    simulator.register_callback(
        Box::new(move |sim| {
            let plots = &mut sim.callback_state;

            // Integrate to get voltage
            let mut v = std::iter::repeat(0.).take(sim.mesh.n_cells-2).collect::<Vec<_>>();
            let mut acc = 0.;
            for i in 0..sim.mesh.n_cells-2 {
                acc += -sim.poisson_state.efield[i].load(Ordering::SeqCst) * sim.mesh.delta_x();
                v[i] = acc;
            }

            let junction_voltage = v.last().unwrap() - v.first().unwrap();
            plots.voltages_over_time.push((sim.time, junction_voltage));

            let current_density_into_left = sim.poisson_state.superparticle_flux_into_left.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor / sim.mesh.cross_section_area;
            let current_density_out_of_left = sim.poisson_state.superparticle_flux_out_of_left.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor / sim.mesh.cross_section_area;
            let current_density_into_right = sim.poisson_state.superparticle_flux_into_right.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor / sim.mesh.cross_section_area;
            let current_density_out_of_right = sim.poisson_state.superparticle_flux_out_of_right.load(Ordering::SeqCst) * -ELECTRON_CHARGE
                * sim.poisson_state.superparticle_factor / sim.mesh.cross_section_area;

            let alpha = (-sim.poisson_solving_interval * decay_per_time).exp();

            last_current_density_into_left    = last_current_density_into_left    * alpha + (1. - alpha) * current_density_into_left;
            last_current_density_out_of_left  = last_current_density_out_of_left  * alpha + (1. - alpha) * current_density_out_of_left;
            last_current_density_into_right   = last_current_density_into_right   * alpha + (1. - alpha) * current_density_into_right;
            last_current_density_out_of_right = last_current_density_out_of_right * alpha + (1. - alpha) * current_density_out_of_right;

            plots.current_density_over_time.push((sim.time, last_current_density_into_left, last_current_density_out_of_left, last_current_density_into_right, last_current_density_out_of_right));

            // Calculate total charge
            let superparticles_in_volume = sim.poisson_state.cell_n_supercarriers.iter().map(|n| {
                n.load(Ordering::SeqCst)
            }).sum::<f64>();
            let superparticles_in_metal_left = sim.poisson_state.n_supercarriers_at_metal_contact.0.load(Ordering::SeqCst);
            let superparticles_in_metal_right = sim.poisson_state.n_supercarriers_at_metal_contact.1.load(Ordering::SeqCst);

            plots.total_charge_over_time.push((
                sim.time,
                superparticles_in_volume * sim.poisson_state.superparticle_factor * -ELECTRON_CHARGE,
                superparticles_in_metal_left * sim.poisson_state.superparticle_factor * -ELECTRON_CHARGE,
                superparticles_in_metal_right * sim.poisson_state.superparticle_factor * -ELECTRON_CHARGE,
            ));
        }),
        units::PS::to_si(0.),
    );

    simulator.simulate(sim_time);

    let mut plots = simulator.callback_state;

    let rho_donors = simulator.semiconductors.iter().map(|s| s.impurity_density * simulator.mesh.volume_of_cell() * ELECTRON_CHARGE).sum::<f64>();

    let plotter_charge_over_time = UnitPlotter::<units::PS, units::ELECTRONS_CHARGES>::new("Charge over time", "t", "\\rho");
    let mut plot_charge_over_time = plotter_charge_over_time.make_plot();
    let trace_volume = plotter_charge_over_time.make_trace(
        plots.total_charge_over_time.iter().map(|&(t, _rho_v, _rho_s_0, _rho_s_1)| t),
        plots.total_charge_over_time.iter().map(|&(_t, rho_v, _rho_s_0, _rho_s_1)| rho_v),
    ).name("Volume");
    plot_charge_over_time.add_trace(trace_volume);
    let trace_donors = plotter_charge_over_time.make_trace(
        plots.total_charge_over_time.iter().map(|&(t, _rho_v, _rho_s_0, _rho_s_1)| t),
        plots.total_charge_over_time.iter().map(|&(_t, _rho_v, _rho_s_0, _rho_s_1)| rho_donors),
    ).name("Impurities");
    plot_charge_over_time.add_trace(trace_donors);
    let trace_surface = plotter_charge_over_time.make_trace(
        plots.total_charge_over_time.iter().map(|&(t, _rho_v, _rho_s_0, _rho_s_1)| t),
        plots.total_charge_over_time.iter().map(|&(_t, _rho_v, rho_s_0, _rho_s_1)| rho_s_0),
    ).name("Surface, left");
    plot_charge_over_time.add_trace(trace_surface);
    let trace_surface = plotter_charge_over_time.make_trace(
        plots.total_charge_over_time.iter().map(|&(t, _rho_v, _rho_s_0, _rho_s_1)| t),
        plots.total_charge_over_time.iter().map(|&(_t, _rho_v, _rho_s_0, rho_s_1)| rho_s_1),
    ).name("Surface, right");
    plot_charge_over_time.add_trace(trace_surface);
    let trace_total = plotter_charge_over_time.make_trace(
        plots.total_charge_over_time.iter().map(|&(t, _rho_v, _rho_s_0, _rho_s_1)| t),
        plots.total_charge_over_time.iter().map(|&(_t, rho_v, rho_s_0, rho_s_1)| rho_v + rho_s_0 + rho_s_1 + rho_donors),
    ).name("Total");
    plot_charge_over_time.add_trace(trace_total);

    // Plot impurity density to carrier_density
    let x = (0..simulator.mesh.n_cells)
        .map(|x| simulator.mesh.idx_to_pos(x));
    let rho_impurity = simulator.semiconductors.iter().map(|sc| ELECTRON_CHARGE * sc.impurity_density);
    let trace = plots.carrier_density.1.make_trace(x, rho_impurity)
        .name("Doping charge").line(Line::new().color("black").dash(DashType::Dash));
    plots.carrier_density.0.add_trace(trace);

    let plotter_voltage_over_time = UnitPlotter::<units::PS, units::VOLT>::new("Voltage over time", "t", "V");
    let mut plot_voltage_over_time = plotter_voltage_over_time.make_plot();
    let trace = plotter_voltage_over_time.make_trace(
        plots.voltages_over_time.iter().map(|&(t, _v)| t),
        plots.voltages_over_time.iter().map(|&(_t, v)| v),
    );
    plot_voltage_over_time.add_trace(trace);

    let plotter_current_over_time = UnitPlotter::<units::PS, units::A_PER_CM2>::new("Current density", "t", "J");
    let mut plot_current_over_time = plotter_current_over_time.make_plot();
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _il, _ol, _ir, _or)| t),
        plots.current_density_over_time.iter().map(|&(_t, il, _ol, _ir, _or)| il),
    )
        .name("Into left contact").line(Line::new().dash(DashType::Dash));
    plot_current_over_time.add_trace(trace);
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _il, _ol, _ir, _or)| t),
        plots.current_density_over_time.iter().map(|&(_t, _il, ol, _ir, _or)| ol),
    )
        .name("Out of left contact").line(Line::new().dash(DashType::Dash));
    plot_current_over_time.add_trace(trace);
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _il, _ol, _ir, _or)| t),
        plots.current_density_over_time.iter().map(|&(_t, _il, _ol, ir, _or)| ir),
    )
        .name("Into right contact").line(Line::new().dash(DashType::Dash));
    plot_current_over_time.add_trace(trace);
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _il, _ol, _ir, _or)| t),
        plots.current_density_over_time.iter().map(|&(_t, _il, _ol, _ir, or)| or),
    )
        .name("Out of right contact").line(Line::new().dash(DashType::Dash));
    plot_current_over_time.add_trace(trace);
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _il, _ol, _ir, _or)| t),
        plots.current_density_over_time.iter().map(|&(_t, il, ol, _ir, _or)| il - ol),
    )
        .name("Total through left contact (ingoing)");
    plot_current_over_time.add_trace(trace);
    let trace = plotter_current_over_time.make_trace(
        plots.current_density_over_time.iter().map(|&(t, _il, _ol, _ir, _or)| t),
        plots.current_density_over_time.iter().map(|&(_t, _il, _ol, ir, or)| or - ir),
    )
        .name("Total through right contact (outgoing)");
    plot_current_over_time.add_trace(trace);

    common::write_plots("poisson", "schottky", [plots.carrier_density.0, plots.efield.0, plots.voltage.0, plot_charge_over_time, plot_voltage_over_time, plot_current_over_time]);

}
