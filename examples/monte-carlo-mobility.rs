#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::EV_TO_J, semiconductor::{Electron, Semiconductor, StepInfo}};
use gambling_simulator::histogram::{Histogram, Binner, Binner2D, UnitBinner, units, DiscreteBinner};

use plotly::{common::{DashType, Line, Mode}, layout::Axis, Layout, Plot, Scatter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use tqdm::tqdm;

use crate::common::write_plots;

struct Histograms {
    velocity: Histogram<Binner2D<UnitBinner<units::KV_PER_CM>, UnitBinner<units::MILLION_CM_PER_SECOND>>>,
    energy: Histogram<Binner2D<UnitBinner<units::KV_PER_CM>, UnitBinner<units::MEV>>>,
    mechanism: Histogram<Binner2D<UnitBinner<units::KV_PER_CM>, DiscreteBinner<&'static str>>>,
    valley: Histogram<Binner2D<UnitBinner<units::KV_PER_CM>, DiscreteBinner<&'static str>>>,
}

fn generate_histogram(
    thread_idx: usize,
    sc: Semiconductor,

    mut histo: Histograms,
    mut step_info: StepInfo, // applied_field overwritten

    n_electrons: usize,
    t_stop: f64,
) -> Histograms {
    let mut rng = ChaCha8Rng::from_os_rng();

    let Γ_valley_idx = sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");

    let steps: Vec<_> = histo.velocity.binner.major.steps_si_and_unit().collect();
    for (efield, _) in tqdm(steps).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        step_info.applied_field = [efield, 0., 0.,];

        for _run in 0..n_electrons {
            let mut electron = Electron::thermalized(&mut rng, &sc, Γ_valley_idx, [0., 0., 0.]);
            let mut t = 0.;

            while t < t_stop {
                // Step electron
                let flight = electron.free_flight(&step_info, &mut rng);
                t += flight.free_flight_time;
                let scatter_mech = electron.scatter(&step_info, &mut rng);

                // Set histogram
                let vx_now = electron.velocity()[0];
                histo.velocity.add((efield, vx_now), t);
                histo.energy.add((efield, electron.energy()), t);

                histo.valley.add((efield, electron.valley().name), t);
                if let Some(mech) = scatter_mech {
                    histo.mechanism.add((efield, mech.name_short), 1.0);
                } // ignore self-scatters
            }
        }
    }
    histo
}

fn main() {
    let sample_sc = Semiconductor::GaAs(300.0);

    let energy_max = 2. * EV_TO_J;

    let step_info = StepInfo {
        applied_field: [0., 0., 0.], // will be overwritten
        maximum_assumed_energy: energy_max,
    };

    let binner_field = UnitBinner::<units::KV_PER_CM>::new(
        0., 30., 60,
    );
    let binner_velocity = UnitBinner::<units::MILLION_CM_PER_SECOND>::new(
        -500., 500., 1000,
    );
    let binner_energy = UnitBinner::<units::MEV>::new_si(
        0., energy_max, 1000,
    );

    let velocity_histo = Histogram::new(
        "velocity".to_string(),
        Binner2D {
            major: binner_field.clone(),
            minor: binner_velocity.clone(),
        },
    );

    let energy_histo = Histogram::new(
        "energy".to_string(),
        Binner2D {
            major: binner_field.clone(),
            minor: binner_energy.clone(),
        },
    );

    let mechanism_names: Vec<&'static str> =
        Electron::all_mechanisms::<ChaCha8Rng>().iter()
            .map(|x| x.name_short)
            .collect();

    let binner_mechanism = DiscreteBinner::new(mechanism_names);
    let mechanism_histo = Histogram::new(
        "Mechanism".into(),
        Binner2D {
            major: binner_field.clone(),
            minor: binner_mechanism.clone(),
        }
    );

    let valley_names: Vec<&'static str> = sample_sc.valleys.iter().map(|v| v.name).collect();
    let binner_valley = DiscreteBinner::new(valley_names);
    let valley_histo = Histogram::new(
        "Valleys".into(),
        Binner2D {
            major: binner_field.clone(),
            minor: binner_valley.clone(),
        }
    );

    let n_electrons = 20;
    let t_stop = 4e-12;
    let n_threads = num_cpus::get();

    let histo: Histograms = std::thread::scope(|scope| {
        let mut histo = Histograms {
            velocity: velocity_histo,
            energy: energy_histo,
            mechanism: mechanism_histo,
            valley: valley_histo,
        };

        let mut handles = (0..n_threads).map(|thread_idx| {
            let sample_sc = sample_sc.clone();
            let histo = Histograms {
                velocity: histo.velocity.get_worker(),
                energy: histo.energy.get_worker(),
                mechanism: histo.mechanism.get_worker(),
                valley: histo.valley.get_worker(),
            };
            let handle = scope.spawn(move || {
                generate_histogram(thread_idx, sample_sc, histo, step_info, n_electrons, t_stop)
            });
            (handle, thread_idx)
        }).collect::<Vec<_>>();

        while !handles.is_empty() {
            let Some(finished_idx) = handles.iter().position(|(h, _idx)| h.is_finished()) else {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            };
            let (handle, thread_idx) = handles.remove(finished_idx);
            let Ok(thread_hist) = handle.join() else {
                eprintln!("thread {thread_idx} panicked :(");
                continue;
            };

            histo.velocity.merge_worker(thread_hist.velocity);
            histo.energy.merge_worker(thread_hist.energy);
            histo.mechanism.merge_worker(thread_hist.mechanism);
            histo.valley.merge_worker(thread_hist.valley);
        }
        histo
    });

    let plot_histo_v = {
        let mut plot_histo_v = Plot::new();

        for (idx, (efield_si, efield_unit)) in binner_field.steps_si_and_unit().enumerate() {
            let histo_v = histo.velocity.as_ref_at_major(efield_si).unwrap();
            let color = common::COLOR_GRADIENT_STANDARD.get(idx as f64 / binner_field.count() as f64);
            let total_time = histo_v.subtotal();

            let trace = Scatter::new(
                    histo_v.all_values().map(|(v_si, _time)| binner_velocity.from_si(v_si)).collect(),
                    histo_v.all_values().map(|(_v_si, time)| time / total_time).collect(),
                )
                .mode(Mode::Lines)
                .name(format!("E_x = {:.3} kV/cm", efield_unit))
                .line(Line::new().color(color));

            plot_histo_v.add_trace(trace);
        }
        plot_histo_v.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title("Velocities")
                .x_axis(
                    Axis::new().title("$v_x [10^6 cm/s]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Time (rel)}$")
                )
        );

        plot_histo_v
    };

    let plot_energy = {
        let mut plot_energy = Plot::new();

        let energy_points: Vec<(f64, f64)> = binner_field.steps_si_and_unit().map(|(efield, _)| {
            let histo_energy = histo.energy.as_ref_at_major(efield).unwrap();
            let mean_energy = histo_energy.mean();

            (efield, mean_energy)
        }).collect();

        let trace = Scatter::new(
                energy_points.iter().map(|(efield, _energy)| binner_field.from_si(*efield)).collect(),
                energy_points.iter().map(|(_efield, energy)| binner_energy.from_si(*energy)).collect(),
            )
            .mode(Mode::Lines);
        plot_energy.add_trace(trace);

        plot_energy.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title("Energy")
                .x_axis(
                    Axis::new().title("$E_x [kV/cm]$")
                )
                .y_axis(
                    Axis::new().title(r"E [meV]")
                )
        );

        plot_energy
    };

    let plot_mobility = {
        let mut plot_mobility = Plot::new();

        let mobility_points = binner_field.steps_si_and_unit().map(|(efield, _)| {
            let histo_v = histo.velocity.as_ref_at_major(efield).unwrap();
            let mean_v = histo_v.mean();

            // from Analysis and simulation of semiconductor devices, 1984 (source for graph in [mixers-and-multipliers-2014])

            let mobility_linear = 0.9; // m/s / (V/m)
            let efield_crit: f64 = 4e5;
            let v_sat = 8.5e4;
            let v_model = (mobility_linear * efield + v_sat * efield.powi(4)/efield_crit.powi(4)) / (1. + efield.powi(4)/efield_crit.powi(4));


            (binner_field.from_si(efield), -binner_velocity.from_si(mean_v), binner_velocity.from_si(v_model))
        }).collect::<Vec<_>>();

        let trace_meas= Scatter::new(
                mobility_points.iter().map(|(v, _meas, _ideal)| *v).collect(),
                mobility_points.iter().map(|(_v, meas, _ideal)| *meas).collect(),
            )
            .mode(Mode::Lines)
            .name("Simulated")
            .line(Line::new().color("blue"));

        let trace_ideal = Scatter::new(
                mobility_points.iter().map(|(v, _meas, _ideal)| *v).collect(),
                mobility_points.iter().map(|(_v, _meas, ideal)| *ideal).collect(),
            )
            .mode(Mode::Lines)
            .name("Reference")
            .line(Line::new().color("orange").dash(DashType::Dot));
        plot_mobility.add_trace(trace_meas);
        plot_mobility.add_trace(trace_ideal);

        plot_mobility.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title("Velocity")
                .x_axis(
                    Axis::new().title("$E_x [kV/cm]$")
                )
                .y_axis(
                    Axis::new().title(r"$\vert v_x\vert [10^6 cm/s]$")
                )
        );
        plot_mobility
    };

    let plot_mechanisms = {
        let mut plot_mechanisms = Plot::new();

        let total_at_field = binner_field.steps_si_and_unit().map(|(field, _)| {
            histo.mechanism.as_ref_at_major(field).unwrap().subtotal()
        }).collect::<Vec<_>>();

        for mech_ty in binner_mechanism.steps() {
            let histo_mech = histo.mechanism.as_ref_at_minor(mech_ty).unwrap();

            let trace = Scatter::new(
                    histo_mech.all_values().map(|(field, _count)| binner_field.from_si(field)).collect(),
                    histo_mech.all_values().zip(total_at_field.iter()).map(|((_field, count), total)| count / total * 100.).collect(),
                )
                .mode(Mode::Lines)
                .name(mech_ty);
            plot_mechanisms.add_trace(trace);
        }

        plot_mechanisms.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title("Mechanisms")
                .x_axis(
                    Axis::new().title("$E_x [kV/cm]$")
                )
                .y_axis(
                    Axis::new().title(r"Percentage of mechanisms")
                )
        );

        plot_mechanisms
    };

    let plot_valley = {
        let mut plot_valley = Plot::new();

        let total_at_field = binner_field.steps_si_and_unit().map(|(field, _)| {
            histo.valley.as_ref_at_major(field).unwrap().subtotal()
        }).collect::<Vec<_>>();

        for valley_name in binner_valley.steps() {
            let histo_valley = histo.valley.as_ref_at_minor(valley_name).unwrap();

            let trace = Scatter::new(
                    histo_valley.all_values().map(|(field, _count)| binner_field.from_si(field)).collect(),
                    histo_valley.all_values().zip(total_at_field.iter()).map(|((_field, count), total)| count / total).collect(),
                )
                .mode(Mode::Lines)
                .name(valley_name);
            plot_valley.add_trace(trace);
        }

        plot_valley.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title("Valley")
                .x_axis(
                    Axis::new().title("$E_x [kV/cm]$")
                )
                .y_axis(
                    Axis::new().title(r"Count, rel, log10")
                )
        );

        plot_valley
    };

    let name = format!("mobility-ni-1e{}", (sample_sc.impurity_density/1e6).log10().round());
    write_plots("monte-carlo", name, [plot_histo_v, plot_energy, plot_mobility, plot_mechanisms, plot_valley]);
}
