#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::{EV_TO_J, J_TO_EV, PLANCK_BAR_SI}, semiconductor::{Electron, Semiconductor, StepInfo}};

use plotly::{common::{Line, Marker}, layout::Axis, Layout, Plot, Scatter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use common::{write_plots, VALLEY_COLORS, Histogram, UnitBinner, DiscreteBinner, Binner2D};
use tqdm::tqdm;

struct Histograms {
    energy_histograms: Vec<Vec<usize>>, // Per valley_idx
    mech_histograms: Vec<Vec<usize>>, // [valley_idx][mech_idx], mech_idx = 0 = scattering, mech_idx+1 = from mechanisms
    velocity_histogram: Vec<usize>,
}
fn generate_histogram(
    thread_idx: usize,
    sc: Semiconductor,
    step_info: StepInfo,

    n_e: usize, // e_max is in step_info
    (v_max, n_v): (f64, usize),

    n_electrons: usize, n_steps: usize,
) -> Histograms {
    let mut rng = ChaCha8Rng::from_os_rng();

    let Γ_valley_idx = sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");
    let mechs = Electron::all_mechanisms::<ChaCha8Rng>();

    let mut energy_histograms: Vec<Vec<usize>> = (0..sc.valleys.len()).map(|_| (0..n_e).map(|_| 0usize).collect()).collect();
    let mut velocity_histogram: Vec<usize> = (0..2*n_v+1).map(|_| 0usize).collect();
    let mut mech_histograms: Vec<Vec<usize>> = (0..sc.valleys.len()).map(|_| (0..mechs.len()).map(|_| 0usize).collect()).collect();

    let energy_histogram_step = step_info.maximum_assumed_energy / n_e as f64;
    let velocity_histogram_step = v_max / n_v as f64;

    for _run in tqdm(0..n_electrons).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        let mut electron = Electron {
            sc: &sc,
            valley_idx: Γ_valley_idx,
            k: [0., 0., 0.,],
            pos: [0., 0., 0.,],
        };
        for _ in 0..n_steps {
            // Set histogram
            let e = electron.energy();
            let histogram_idx = (e / energy_histogram_step).round() as usize;
            if histogram_idx < energy_histograms[electron.valley_idx].len() {
                energy_histograms[electron.valley_idx][histogram_idx] += 1;
            }

            let v = electron.velocity();
            let histogram_idx = ((v[0] / velocity_histogram_step).round() + n_v as f64).floor();
            if histogram_idx >= 0. && histogram_idx < velocity_histogram.len() as f64 {
                velocity_histogram[histogram_idx as usize] += 1;
            }

            // Step electron
            electron.free_flight(&step_info, &mut rng);
            let mut mech_idx = 0;
            if let Some(mech) = electron.scatter(&step_info, &mut rng) {
                // Find index of mechanism
                mech_idx = 1 + mechs.iter().position(|other| other.name_short == mech.name_short).expect("Unknown mechanism");
            }
            mech_histograms[electron.valley_idx][mech_idx] += 1;
        }
    }
    Histograms { energy_histograms, mech_histograms, velocity_histogram }
}

fn main() {
    let sample_sc = Semiconductor::GaAs(300.0);
    let Γ_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");

    let energy_max = 2. * EV_TO_J;
    let e_x = 4.; // kV/cm

    let step_info = StepInfo {
        applied_field: [e_x * 1e3 * 1e2, 0., 0.],
        maximum_assumed_energy: energy_max,
    };

    let energy_histogram_step = 1e-3 * EV_TO_J;
    let n_e = (energy_max / energy_histogram_step).ceil() as usize;

    let k_at_emax = sample_sc.valleys[Γ_valley_idx].kmag_for_e(energy_max);
    let v_at_emax = PLANCK_BAR_SI / sample_sc.valleys[Γ_valley_idx].effective_mass() * k_at_emax;
    let v_step = v_at_emax / 1000.;
    let n_v = (v_at_emax / v_step).ceil() as usize;

    let n_electrons = 100;
    let n_steps = 50000;
    let n_threads = num_cpus::get();
    let n_points = n_electrons * n_steps * n_threads;

    let histograms: Histograms = std::thread::scope(|scope| {
        let mut histograms: Option<Histograms> = None;
        let mut handles = (0..n_threads).map(|thread_idx| {
            let sample_sc = sample_sc.clone();
            let handle = scope.spawn(move || generate_histogram(thread_idx, sample_sc, step_info, n_e, (v_at_emax, n_v), n_electrons, n_steps));
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
                let old_energy_histograms = std::mem::replace(&mut h.energy_histograms, Vec::new());
                h.energy_histograms = old_energy_histograms.into_iter()
                    .zip(thread_hist.energy_histograms)
                    .map(|(a, b)| a.into_iter().zip(b.into_iter()).map(|(ai, bi)| ai + bi).collect())
                    .collect();
                let old_mech_histograms = std::mem::replace(&mut h.mech_histograms, Vec::new());
                h.mech_histograms = old_mech_histograms.into_iter()
                    .zip(thread_hist.mech_histograms)
                    .map(|(a, b)| a.into_iter().zip(b.into_iter()).map(|(ai, bi)| ai + bi).collect())
                    .collect();
                let old_velocity_histogram = std::mem::replace(&mut h.velocity_histogram, Vec::new());
                h.velocity_histogram = old_velocity_histogram.into_iter()
                    .zip(thread_hist.velocity_histogram)
                    .map(|(a, b)| a + b)
                    .collect();
            } else {
                histograms = Some(thread_hist);
            };
        }
        histograms.unwrap()
    });

    let mechs = Electron::all_mechanisms::<ChaCha8Rng>();

    let integrated_v = histograms.velocity_histogram.iter().enumerate().map(|(idx, count)| {
        let v = (idx as f64 - n_v as f64) * v_step;
        v * *count as f64
    }).sum::<f64>();
    let mean_v = integrated_v / n_points as f64;
    let integrated_v_stddev = histograms.velocity_histogram.iter().enumerate().map(|(idx, count)| {
        let v = (idx as f64 - n_v as f64) * v_step;
        (v - mean_v).powi(2) * *count as f64
    }).sum::<f64>();
    let mean_v_stddev = (integrated_v_stddev / n_points as f64).sqrt();
    eprintln!("Mean velocty = {:.4} ± {:.4} 10^7 cm/s", mean_v / 1e5, mean_v_stddev / 1e5);
    let gaas_mobility = 0.9; // m/s / (V/m)
    let lin_mean_vel = -gaas_mobility * step_info.applied_field[0];
    eprintln!("  (should be {:.4} 10^7 cm/s)", lin_mean_vel / 1e5);

    let mut plot_histo_e = Plot::new();
    for ((valley_idx, valley), energy_histogram) in sample_sc.valleys.iter().enumerate().zip(histograms.energy_histograms) {
        let histo_e = Scatter::new(
                (0..energy_histogram.len()).map(|x| x as f64 * energy_histogram_step * J_TO_EV * 1e3).collect(),
                energy_histogram.iter().map(|&count| count as f64 / n_points as f64 / (energy_histogram_step * J_TO_EV * 1e3)).collect(),
            )
            .name(valley.name)
            .line(Line::new().color(VALLEY_COLORS[valley_idx]));
        plot_histo_e.add_trace(histo_e);
    }

    let mut plot_histo_v = Plot::new();
    let histo_v = Scatter::new(
            (0..histograms.velocity_histogram.len()).map(|x| (x as f64 - n_v as f64) * v_step / 1e6).collect(),
            histograms.velocity_histogram.iter().map(|&count| count as f64 / n_points as f64 / (v_step / 1e6)).collect(),
        );
    plot_histo_v.add_trace(histo_v);

    let mut plot_histo_mechs = Plot::new();
    for ((valley_idx, valley), mech_histogram) in sample_sc.valleys.iter().enumerate().zip(histograms.mech_histograms) {
        let histo_mech = Scatter::new(
                (0..mech_histogram.len()).map(|x| x as f64).collect(),
                mech_histogram.iter().map(|&count| (count as f64 / n_points as f64).log10()).collect(),
            )
            .mode(plotly::common::Mode::Markers)
            .name(valley.name)
            .text_array(mech_histogram.iter().map(|&count| format!("{:.2}%", 100. * count as f64 / n_points as f64)).collect::<Vec<_>>())
            .marker(Marker::new().size(10).color(VALLEY_COLORS[valley_idx]));

        plot_histo_mechs.add_trace(histo_mech);
    }
    let mut names = mechs.iter().map(|mech| mech.name_short).collect::<Vec<_>>();
    names.insert(0, "self-scatter");

    let n_y_log10 = 5;

    plot_histo_mechs.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title(format!(r"$\text{{Mechanisms}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
            .x_axis(
                Axis::new().title("Mechanism")
                    .tick_values((0..mechs.len()+1).map(|x| x as f64).collect())
                    .tick_text(names)
            )
            .y_axis(
                Axis::new().title(r"$\text{Count (rel)}$")
                    .range(vec![-n_y_log10, 0])
                    .tick_values((-n_y_log10..=0).map(|x| x as f64).collect())
                    .tick_text((-n_y_log10..=0).map(|e| format!("$10^{{{e}}}$")).collect())
            )
    );

    plot_histo_e.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title(format!(r"$\text{{Energies}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
            .x_axis(
                Axis::new().title("$E [meV]$")
            )
            .y_axis(
                Axis::new().title(r"$\text{Count (rel)} 1/meV$")
            )
    );

    plot_histo_v.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title(format!(r"$\text{{Velocities}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
            .x_axis(
                Axis::new().title("$v_x [Mm/s]$")
            )
            .y_axis(
                Axis::new().title(r"$\text{Count (rel)} 1/(Mm/s)$")
            )
    );

    write_plots("monte-carlo", "statistics", [plot_histo_e, plot_histo_v, plot_histo_mechs]);
}
