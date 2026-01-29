#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::EV_TO_J, semiconductor::{Electron, Semiconductor, StepInfo}};

use plotly::{color::Rgb, common::{DashType, Line, Marker, Mode}, layout::Axis, Layout, Plot, Scatter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use common::write_plots;
use tqdm::tqdm;

struct Histograms {
    velocity_histograms: Vec<Vec<usize>>, // [field_idx][vel_idx] (vel_idx = n_velocity is zero-velocity, 2*n_velocity+1 items)
}
impl Histograms {
    fn combine(&mut self, other: Histograms) {
        for (my_hist, other_hist) in self.velocity_histograms.iter_mut().zip(other.velocity_histograms.into_iter()) {
            for (my_item, other_item) in my_hist.iter_mut().zip(other_hist.into_iter()) {
                *my_item += other_item;
            }
        }
    }
}

fn generate_histogram(
    thread_idx: usize,
    sc: Semiconductor,
    mut step_info: StepInfo, // applied_field overwritten

    (efield_max, n_efield): (f64, usize),
    (velocity_max, n_velocity): (f64, usize),

    n_electrons: usize,
    n_steps: usize,
) -> Histograms {
    let mut rng = ChaCha8Rng::from_os_rng();

    let Γ_valley_idx = sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");

    let mut velocity_histograms: Vec<Vec<usize>> = (0..n_efield).map(|_| (0..2*n_velocity+1).map(|_| 0usize).collect()).collect();
    let velocity_histogram_step = velocity_max / n_velocity as f64;

    for efield_idx in tqdm(0..n_efield).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        let efield = efield_max * efield_idx as f64 / n_efield as f64;
        step_info.applied_field = [efield, 0., 0.,];

        for _run in 0..n_electrons {
            let mut electron = Electron {
                sc: &sc,
                valley_idx: Γ_valley_idx,
                k: [0., 0., 0.,],
                pos: [0., 0., 0.,],
            };
            for _ in 0..n_steps {
                // Set histogram
                let v = electron.velocity();
                let histogram_idx = ((v[0] / velocity_histogram_step).round() + n_velocity as f64).floor();
                if histogram_idx >= 0. && histogram_idx < velocity_histograms[efield_idx].len() as f64 {
                    velocity_histograms[efield_idx][histogram_idx as usize] += 1;
                }

                // Step electron
                electron.free_flight(&step_info, &mut rng);
                electron.scatter(&step_info, &mut rng);
            }
        }
    }
    Histograms { velocity_histograms }
}

fn main() {
    let sample_sc = Semiconductor::GaAs(300.0);

    let energy_max = 2. * EV_TO_J;
    let e_x_max = 10.; // kV/cm
    let efield_max = e_x_max * 1e3 * 1e2; // V/m

    let step_info = StepInfo {
        applied_field: [0., 0., 0.], // will be overwritten
        maximum_assumed_energy: energy_max,
    };

    // good bounds for velocity
    // let k_at_emax = sample_sc.valleys[Γ_valley_idx].kmag_for_e(energy_max);
    // let v_at_emax = PLANCK_SI / sample_sc.valleys[Γ_valley_idx].effective_mass() * k_at_emax;
    let v_at_emax = 20.0e5f64;
    let v_step = v_at_emax / 100.;
    let n_v = (v_at_emax / v_step).ceil() as usize;

    let n_electrons = 10;
    let n_steps = 5000;
    let n_efields = 60;
    let n_threads = num_cpus::get();
    let n_points = n_electrons * n_steps * n_threads;

    let histograms: Histograms = std::thread::scope(|scope| {
        let mut histograms: Option<Histograms> = None;
        let mut handles = (0..n_threads).map(|thread_idx| {
            let sample_sc = sample_sc.clone();
            let handle = scope.spawn(move || {
                generate_histogram(thread_idx, sample_sc, step_info, (efield_max, n_efields), (v_at_emax, n_v), n_electrons, n_steps)
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

            if let Some(ref mut h) = histograms {
                // TODO: This is not particularily efficient lmao
                h.combine(thread_hist);
            } else {
                histograms = Some(thread_hist);
            };
        }
        histograms.unwrap()
    });

    let mut plot_histo_v = Plot::new();
    let start_color = [0., 0., 255.];
    let end_color = [255., 128., 0.];

    let bump_size = 1.; // kV/cm / (rel-count/10^7cm/s)
    for efield_idx in 0..n_efields {
        let histo = &histograms.velocity_histograms[efield_idx];
        let efield = efield_max * efield_idx as f64 / n_efields as f64;

        let color = [
            start_color[0] + (end_color[0] - start_color[0]) * (efield_idx as f64 / n_efields as f64),
            start_color[1] + (end_color[1] - start_color[1]) * (efield_idx as f64 / n_efields as f64),
            start_color[2] + (end_color[2] - start_color[2]) * (efield_idx as f64 / n_efields as f64),
        ];
        let color = Rgb::new(color[0] as u8, color[1] as u8, color[2] as u8);

        let histo_v = Scatter::new(
                (0..histo.len()).map(|idx| (idx as f64 - n_v as f64) * v_step / 1e5).collect(),
                histo.iter().map(|&count| count as f64 / n_points as f64 / (v_step / 1e6) * bump_size + efield / 1e5).collect(),
            )
            .mode(Mode::Lines)
            .line(Line::new().color(color));

        let integrated_v = histo.iter()
            .enumerate()
            .map(|(idx, &count)| (idx as f64 - n_v as f64) * v_step / 1e5 * count as f64)
            .sum::<f64>();
        let mean_v = integrated_v / histo.iter().sum::<usize>() as f64;
        let max_count = histo.iter().map(|&count| count as f64 / n_points as f64 / (v_step / 1e6) * bump_size + efield / 1e5).max_by(f64::total_cmp).unwrap();

        let point = Scatter::new(
                vec![mean_v], vec![max_count],
            )
            .mode(Mode::Markers)
            .marker(Marker::new().size(10).color(color));

        plot_histo_v.add_trace(histo_v);
        plot_histo_v.add_trace(point);
    }
    plot_histo_v.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title("Velocities")
            .x_axis(
                Axis::new().title("$v_x [10^7 cm/s]$")
            )
            .y_axis(
                Axis::new().title("$E_x [kV/cm]$")
            )
    );

    let ideal_mobility = 6500.0; // cm² / Vs
    let mut plot_velocity = Plot::new();
    let mobility_points = histograms.velocity_histograms.iter().enumerate().map(|(efield_idx, velocity_histogram)| {
        let efield = efield_max * efield_idx as f64 / n_efields as f64;

        let integrated_v = velocity_histogram.iter()
            .enumerate()
            .map(|(idx, &count)| (idx as f64 - n_v as f64) * v_step * count as f64)
            .sum::<f64>();
        let mean_v = integrated_v / velocity_histogram.iter().sum::<usize>() as f64;

        // rough fit from https://www.ioffe.ru/SVA/NSM/Semicond/GaAs/Figs/437.gif
        let ideal_v_lin = ideal_mobility/1e4 * efield; // from bulk low-field mobility
        let ideal_v = if efield < 2.0e5 {
            ideal_v_lin
        } else {
            let ideal_v_sat = 1.2e5 * (1. + (-(efield - 4.0e5)/1.0e5).exp());
            let alpha = -4.0;
            (ideal_v_lin.powf(alpha) + ideal_v_sat.powf(alpha)).powf(1. / alpha)
        };

        (efield / 1e5, -mean_v / 1e5, ideal_v / 1e5) // 10^7 cm/s
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
