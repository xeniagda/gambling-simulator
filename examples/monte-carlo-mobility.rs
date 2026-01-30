#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::EV_TO_J, semiconductor::{Electron, Semiconductor, StepInfo}};

use plotly::{common::{DashType, Line, Mode}, layout::Axis, Layout, Plot, Scatter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use common::{write_plots, Binner, Binner2D, Histogram, UnitBinner};
use tqdm::tqdm;

type VelocityHistogram = Histogram<Binner2D<UnitBinner, UnitBinner>>; // y = field strength, x = velocity

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


    for efield in tqdm(histo.binner.bx.items()).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        step_info.applied_field = [efield, 0., 0.,];

        for _run in 0..n_electrons {
            let mut electron = Electron::thermalized(&mut rng, &sc, Γ_valley_idx, [0., 0., 0.]);
            let mut t = 0.;

            while t < t_stop {
                // Step electron
                let vx_previous = electron.velocity()[0];
                let flight = electron.free_flight(&step_info, &mut rng);
                t += flight.free_flight_time;
                electron.scatter(&step_info, &mut rng);

                // Set histogram
                let vx_now = electron.velocity()[0];

                let n = 100;
                for i in 0..n {
                    let alpha = i as f64 / n as f64;
                    let vx = vx_previous * alpha + vx_now * (1. - alpha);
                    histo.add_value((efield, vx), t);
                }
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

    let histo = VelocityHistogram::new(
        "velocity".to_string(),
        Binner2D {
            bx: UnitBinner {
                unit: common::UNIT_KV_CM,
                start_unit: 0.,
                end_unit: 30.,
                count: 60,
            },
            by: UnitBinner {
                unit: common::UNIT_10_7_CM_S,
                start_unit: -20.,
                end_unit: 20.,
                count: 200,
            },
        }
    );

    let n_electrons = 40;
    let t_stop = 4e-12;
    let n_threads = num_cpus::get();

    let histo: VelocityHistogram = std::thread::scope(|scope| {
        let mut histo = histo;

        let mut handles = (0..n_threads).map(|thread_idx| {
            let sample_sc = sample_sc.clone();
            let histo = histo.new_worker();
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

    let mut plot_histo_v = Plot::new();

    for (efield_idx, efield) in histo.binner.bx.items().into_iter().enumerate() {
        let n_efields = histo.binner.bx.count();

        let color = common::COLOR_GRADIENT_STANDARD.get(efield_idx as f64 / n_efields as f64);

        let vels = histo.binner.by.items();

        let total_for_field = vels.iter().map(|&vel| histo.get_count((efield, vel))).sum::<f64>();

        let histo_v = Scatter::new(
                vels.iter().map(|&x| x / histo.binner.by.unit.1).collect(),
                vels.iter().map(|&vel| histo.get_count((efield, vel)) / total_for_field).collect(),
            )
            .mode(Mode::Lines)
            .name(format!("E_x = {:.3} kV/cm", efield / histo.binner.bx.unit.1))
            .line(Line::new().color(color));

        plot_histo_v.add_trace(histo_v);
    }
    plot_histo_v.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title("Velocities")
            .x_axis(
                Axis::new().title("$v_x [10^7 cm/s]$")
            )
            // .y_axis(
            //     Axis::new().title("$E_x [kV/cm]$")
            // )
    );

    let ideal_mobility = 6500.0; // cm² / Vs
    let mut plot_velocity = Plot::new();
    let mobility_points = histo.binner.bx.items().into_iter().map(|efield| {
        let vels = histo.binner.by.items();

        let integrated_v = vels.iter()
            .map(|&vel| vel * histo.get_count((efield, vel)))
            .sum::<f64>();
        let total_time = vels.iter()
            .map(|&vel| histo.get_count((efield, vel)))
            .sum::<f64>();

        let mean_v = integrated_v / total_time;

        // rough fit from https://www.ioffe.ru/SVA/NSM/Semicond/GaAs/Figs/437.gif
        let ideal_v_lin = ideal_mobility/1e4 * efield; // from bulk low-field mobility
        let ideal_v = if efield < 2.0e5 {
            ideal_v_lin
        } else {
            let ideal_v_sat = 1.2e5 * (1. + (-(efield - 4.0e5)/1.0e5).exp());
            let alpha = -4.0;
            (ideal_v_lin.powf(alpha) + ideal_v_sat.powf(alpha)).powf(1. / alpha)
        };

        (efield / 1e5, -mean_v / histo.binner.bx.unit.1, ideal_v / histo.binner.bx.unit.1) // 10^7 cm/s
    }).collect::<Vec<_>>();

    let mobility_trace = Scatter::new(
            mobility_points.iter().map(|&(x, _v, _v_id)| x).collect(),
            mobility_points.iter().map(|&(_x, v, _v_id)| v).collect(),
        )
        .mode(Mode::Lines)
        .name("Simulation");
    plot_velocity.add_trace(mobility_trace);

    let ideal_trace = Scatter::new(
            mobility_points.iter().map(|&(x, _v, _v_id)| x).collect(),
            mobility_points.iter().map(|&(_x, _v, v_id)| v_id).collect(),
        )
        .mode(Mode::Lines)
        .line(Line::new().dash(DashType::Dot))
        .name("Ref.");
    plot_velocity.add_trace(ideal_trace);

    plot_velocity.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title("Velocities")
            .x_axis(
                Axis::new().title("$E_x [kV/cm]$")
            )
            .y_axis(
                Axis::new().title(r"$\vert v_x \vert [10^7 cm/s]$")
            )
    );

    write_plots("monte-carlo", "mobility", [plot_histo_v, plot_velocity]);
}
