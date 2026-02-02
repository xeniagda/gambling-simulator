#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::EV_TO_J, semiconductor::{Electron, Semiconductor, StepInfo}};
use gambling_simulator::histogram::{Histogram, Binner, Binner2D, UnitBinner, units};

use plotly::{common::{DashType, Line, Mode}, layout::Axis, Layout, Plot, Scatter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use tqdm::tqdm;

use crate::common::write_plots;

type VelocityHistogram = Histogram<Binner2D<UnitBinner<units::KV_PER_CM>, UnitBinner<units::TEN_MILLION_CM_PER_SECOND>>>; // y = field strength, x = velocity

fn generate_histogram(
    thread_idx: usize,
    sc: Semiconductor,

    mut histo: VelocityHistogram,
    mut step_info: StepInfo, // applied_field overwritten

    n_electrons: usize,
    t_stop: f64,
) -> VelocityHistogram {
    let mut rng = ChaCha8Rng::from_os_rng();

    let Γ_valley_idx = sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");

    let steps: Vec<_> = histo.binner.major.steps_si_and_unit().collect();
    for (efield, _) in tqdm(steps).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        step_info.applied_field = [efield, 0., 0.,];

        for _run in 0..n_electrons {
            let mut electron = Electron::thermalized(&mut rng, &sc, Γ_valley_idx, [0., 0., 0.]);
            let mut t = 0.;

            while t < t_stop {
                // Step electron
                let flight = electron.free_flight(&step_info, &mut rng);
                t += flight.free_flight_time;
                electron.scatter(&step_info, &mut rng);

                // Set histogram
                let vx_now = electron.velocity()[0];
                histo.add((efield, vx_now), t);
            }
        }
    }
    histo
}

fn main() {
    let mut sample_sc = Semiconductor::GaAs(300.0);
    sample_sc.impurity_density = 1e17 * 1e6;

    let energy_max = 2. * EV_TO_J;

    let step_info = StepInfo {
        applied_field: [0., 0., 0.], // will be overwritten
        maximum_assumed_energy: energy_max,
    };

    let binner_field = UnitBinner::<units::KV_PER_CM>::new(
        0., 30., 60,
    );

    let binner_velocity = UnitBinner::<units::TEN_MILLION_CM_PER_SECOND>::new(
        -50., 50., 1000,
    );

    let histo = VelocityHistogram::new(
        "velocity".to_string(),
        Binner2D {
            major: binner_field.clone(),
            minor: binner_velocity.clone(),
        },
    );

    let n_electrons = 1000;
    let t_stop = 4e-12;
    let n_threads = num_cpus::get();

    let histo: VelocityHistogram = std::thread::scope(|scope| {
        let mut histo = histo;

        let mut handles = (0..n_threads).map(|thread_idx| {
            let sample_sc = sample_sc.clone();
            let histo = histo.get_worker();
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

            histo.merge_worker(thread_hist);
        }
        histo
    });

    let plot_histo_v = {
        let mut plot_histo_v = Plot::new();

        for (idx, (efield_si, efield_unit)) in binner_field.steps_si_and_unit().enumerate() {
            let histo_v = histo.as_ref_at_major(efield_si).unwrap();
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
                    Axis::new().title("$v_x [10^7 cm/s]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Time (rel)}$")
                )
        );

        plot_histo_v
    };

    let plot_mobility = {
        let mut plot_mobility = Plot::new();
        let ideal_mobility = 6500.0; // cm² / Vs

        let mobility_points = binner_field.steps_si_and_unit().map(|(efield_si, _)| {
            let histo_v = histo.as_ref_at_major(efield_si).unwrap();
            let mean_v = histo_v.mean();

            // rough fit from https://www.ioffe.ru/SVA/NSM/Semicond/GaAs/Figs/437.gif
            let ideal_v_lin = ideal_mobility/1e4 * efield_si; // from bulk low-field mobility
            let ideal_v = if efield_si < 2.0e5 {
                ideal_v_lin
            } else {
                let ideal_v_sat = 1.2e5 * (1. + (-(efield_si - 4.0e5)/1.0e5).exp());
                let alpha = -4.0;
                (ideal_v_lin.powf(alpha) + ideal_v_sat.powf(alpha)).powf(1. / alpha)
            };

            (binner_field.from_si(efield_si), -binner_velocity.from_si(mean_v), binner_velocity.from_si(ideal_v))
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
                    Axis::new().title(r"$\vert v_x\vert [10^7 cm/s]$")
                )
        );
        plot_mobility
    };

    write_plots("monte-carlo", "mobility", [plot_histo_v, plot_mobility]);
}
