use std::sync::Arc;

use gambling_simulator::consts::{ELECTRON_CHARGE, EPS0};
use gambling_simulator::semiconductor::{Electron, Semiconductor, StepInfo};
use gambling_simulator::units::{self, EV, NM, PER_CM_CUBED, PS, Unit};
use gambling_simulator::histogram::{Binner, Histogram, UnitBinner};

use plotly::Plot;
use plotly::common::Line;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod common;
use common::write_plots;

const N_CELLS: usize = 100;
const N_ELECTRONS: usize = 10000;

fn main() {
    let simulation_time = PS::to_si(75.);
    let plot_interval = PS::to_si(5.);

    let poisson_interval = PS::to_si(0.02);

    let maximum_assumed_energy = EV::to_si(2.);

    let mut rng = ChaCha8Rng::from_os_rng();

    let temperature = 300.;

    let cross_area: f64 = NM::to_si(2000.,).powi(2); // 1um^2

    // for a 1e18cm^-3 doped NI-junction we expect the space charge region to be about 30nm long
    // let's make the computational domain 300nm long with the junction in the center
    let binner_x: UnitBinner<NM> = UnitBinner::new(
        "x",
        -4000., 4000., N_CELLS,
    );

    let doping_level_1 = PER_CM_CUBED::to_si(1.16e15);
    let doping_level_2 = PER_CM_CUBED::to_si(2.41e13);

    let total_electrons = cross_area * (doping_level_1 * -binner_x.start_si + doping_level_2 * binner_x.end_si);

    eprintln!("Number of electrons in physical domain: 10^{:.1}", total_electrons.log10());
    eprintln!("Number of electrons simulated: 10^{:.1}", (N_ELECTRONS as f64).log10());
    eprintln!("Each simulated electron represents {:.1} dBe⁻", 20. * (total_electrons/N_ELECTRONS as f64).log10());

    let scs = binner_x.steps().map(|x| {
        let mut sc = Semiconductor::GaAs(temperature);
        sc.impurity_density = if x < 0. { doping_level_1 } else { doping_level_2 };
        Arc::new(sc)
    }).collect::<Vec<_>>();

    let mut electrons: Vec<Electron> = Vec::new();
    for _ in 0..N_ELECTRONS {
        let x = if rng.random::<f64>() < doping_level_1 / (doping_level_1 + doping_level_2) {
            rng.random_range((binner_x.start_si - binner_x.bin_size()/2.) .. 0.)
        } else {
            rng.random_range(0. .. (binner_x.end_si + binner_x.bin_size()/2.))
        };

        let i = binner_x.bin(x).unwrap();
        electrons.push(Electron::thermalized(
            &mut rng, scs[i].clone(), 0, [x, 0., 0.,], [0., 0., 0.,]
        ));
    }

    let mut charge_histo: Histogram<UnitBinner<NM>> = Histogram::new(
        "charge total".into(), binner_x.clone(),
    );
    let mut efield_histo: Histogram<UnitBinner<NM>> = Histogram::new(
        "efield".into(), binner_x.off_grid_grow(),
    );
    let mut voltage_histo: Histogram<UnitBinner<NM>> = Histogram::new(
        "voltage".into(), binner_x.off_grid_grow().off_grid_grow(),
    );

    let mut charge_histo_int: Histogram<UnitBinner<NM>> = Histogram::new(
        "charge total".into(), binner_x.clone(),
    );
    let mut efield_histo_int: Histogram<UnitBinner<NM>> = Histogram::new(
        "efield".into(), binner_x.off_grid_grow(),
    );
    let mut voltage_histo_int: Histogram<UnitBinner<NM>> = Histogram::new(
        "voltage".into(), binner_x.off_grid_grow().off_grid_grow(),
    );

    let mut plot_charge = Plot::new();
    let mut plot_efield = Plot::new();
    let mut plot_voltage = Plot::new();

    common::plot_histogram::set_default_layout(&mut plot_charge, "Charge");
    common::plot_histogram::set_default_layout(&mut plot_efield, "Field");
    common::plot_histogram::set_default_layout(&mut plot_voltage, "Voltage");

    common::plot_histogram::quantity::set_layout::<_, units::ELECTRONS_PER_CM_CUBED>(&mut plot_charge, charge_histo_int.as_ref(), r"\rho");
    common::plot_histogram::quantity::set_layout::<_, units::KV_PER_CM>(&mut plot_efield, efield_histo_int.as_ref(), r"E");
    common::plot_histogram::quantity::set_layout::<_, units::MILLIVOLT>(&mut plot_voltage, voltage_histo_int.as_ref(), r"V");

    // at each poisson step all electrons need to be interrupted in their flight
    // this vector stores the amount of remaining time for the electron post-flight
    let mut time_left_in_free_flight: Vec<_> = electrons.iter().map(|_| 0.).collect();

    let wall_time_started = std::time::Instant::now();

    let mut last_plot_at = 0.;
    let count = (simulation_time/poisson_interval) as usize;
    let mut tqdm = tqdm::tqdm(0..count);
    for step in 0..count {
        tqdm.next();
        let t0 = step as f64 * poisson_interval;
        let elapsed = wall_time_started.elapsed().as_secs_f64();
        tqdm.set_desc(Some(format!("t = {}, {:.3} ps/s ({:.1} dBs/s)", units::PS::format(t0), units::PS::from_si(t0) / elapsed, (t0/elapsed).log10()*10.)));

        // calculate voltages in each cell

        let simulated_electron_charge = ELECTRON_CHARGE * total_electrons / N_ELECTRONS as f64;
        let steps_per_plot = plot_interval / poisson_interval;
        charge_histo.reset();
        efield_histo.reset();
        voltage_histo.reset();
        for e in &electrons {
            let delta_vol = cross_area * binner_x.bin_size();
            charge_histo.add(e.pos[0], -simulated_electron_charge / delta_vol);
            charge_histo_int.add(e.pos[0], -simulated_electron_charge / delta_vol / steps_per_plot);
        }
        for i in 0..N_CELLS {
            let x = binner_x.unbin(i).unwrap();
            charge_histo.add(x, scs[i].impurity_density * ELECTRON_CHARGE);
            charge_histo_int.add(x, scs[i].impurity_density * ELECTRON_CHARGE / steps_per_plot);
        }

        // poisson solver
        // dE/dx = rho / epsilon
        for i in 0..N_CELLS {
            let x = binner_x.unbin(i).unwrap();
            let ex_prev = efield_histo.get(x - binner_x.bin_size()/2.);
            let ex_next = ex_prev + charge_histo.get(x) * binner_x.bin_size() / (EPS0 * scs[i].relative_dielectric_static);
            efield_histo.add(x + binner_x.bin_size()/2., ex_next);
        }


        // dV/dx = -E
        for i in 0..N_CELLS {
            let x = binner_x.off_grid_grow().unbin(i).unwrap();
            let v_prev = voltage_histo.get(x - binner_x.bin_size()/2.);
            let v_next = v_prev - efield_histo.get(x) * binner_x.bin_size();
            voltage_histo.add(x + binner_x.bin_size()/2., v_next);
        }

        if t0 - last_plot_at > plot_interval {
            last_plot_at = t0;
            for i in 0..N_CELLS {
                let x = binner_x.unbin(i).unwrap();
                let ex_prev = efield_histo_int.get(x - binner_x.bin_size()/2.);
                let ex_next = ex_prev + charge_histo_int.get(x) * binner_x.bin_size() / (EPS0 * scs[i].relative_dielectric_static);
                efield_histo_int.add(x + binner_x.bin_size()/2., ex_next);
            }

            // dV/dx = -E
            for i in 0..N_CELLS {
                let x = binner_x.off_grid_grow().unbin(i).unwrap();
                let v_prev = voltage_histo_int.get(x - binner_x.bin_size()/2.);
                let v_next = v_prev - efield_histo_int.get(x) * binner_x.bin_size();
                voltage_histo_int.add(x + binner_x.bin_size()/2., v_next);
            }

            let trace = common::plot_histogram::quantity::plot::<_, units::ELECTRONS_PER_CM_CUBED>(charge_histo_int.as_ref());
            plot_charge.add_trace(
                trace
                    .name(format!("t = {} ps", PS::from_si(t0)))
                    .line(Line::new().color(common::COLOR_GRADIENT_STANDARD.get(t0 / simulation_time)))
            );
            let trace = common::plot_histogram::quantity::plot::<_, units::KV_PER_CM>(efield_histo_int.as_ref());
            plot_efield.add_trace(
                trace
                    .name(format!("t = {} ps", PS::from_si(t0)))
                    .line(Line::new().color(common::COLOR_GRADIENT_STANDARD.get(t0 / simulation_time)))
            );
            let trace = common::plot_histogram::quantity::plot::<_, units::MILLIVOLT>(voltage_histo_int.as_ref());
            plot_voltage.add_trace(
                trace
                    .name(format!("t = {} ps", PS::from_si(t0)))
                    .line(Line::new().color(common::COLOR_GRADIENT_STANDARD.get(t0 / simulation_time)))
            );

            charge_histo_int.reset();
            efield_histo_int.reset();
            voltage_histo_int.reset();
        }

        // step each electron
        for i in 0..electrons.len() {

            let electron = &mut electrons[i];
            let time_left_in_free_flight = &mut time_left_in_free_flight[i];

            let mut t = 0.;

            loop {
                // Boundary conditions
                let Ok(idx) = binner_x.bin(electron.pos[0]) else {
                    // electron left the confines of the simulator
                    if electron.pos[0] > binner_x.end_si + binner_x.bin_size()/2. {
                        // replace electron with a new electron coming from bulk of the SC
                        // but pick electron with velocity from the right way
                        let field = [efield_histo.get(binner_x.end_si), 0., 0.,];
                        loop {
                            *electron = Electron::thermalized(&mut rng, scs[scs.len()-1].clone(), 0, [binner_x.end_si+binner_x.bin_size()/2., 0., 0.], field);
                            if electron.k[0] < 0. {
                                break;
                            }
                        }
                    } else if electron.pos[0] < binner_x.start_si-binner_x.bin_size()/2. {
                        let field = [efield_histo.get(binner_x.start_si), 0., 0.,];
                        loop {
                            *electron = Electron::thermalized(&mut rng, scs[0].clone(), 0, [binner_x.start_si-binner_x.bin_size()/2., 0., 0.], field);
                            if electron.k[0] > 0. {
                                break;
                            }
                        }
                    } else {
                        panic!("Electron left binner range but still inside semiconductor? At {:?}", electron.pos);
                    }
                    continue;
                };
                // sync with whatever cell we're in
                electron.sc = scs[idx].clone();

                let info = StepInfo {
                    applied_field: [efield_histo.get(electron.pos[0]), 0., 0.,],
                    maximum_assumed_energy,
                    scattering_mechanisms: Semiconductor::all_mechanisms::<ChaCha8Rng>(),
                };

                if t + *time_left_in_free_flight > poisson_interval {
                    // do last half-flight and then break
                    let dt = poisson_interval - t;
                    electron.free_flight(dt, &info);
                    *time_left_in_free_flight -= dt;
                    break;
                }
                // we have enough time to perform the whole remaining of the free flight
                electron.free_flight(*time_left_in_free_flight, &info);
                t += *time_left_in_free_flight;

                // scatter and start next flight
                electron.scatter(&info, &mut rng);
                *time_left_in_free_flight = electron.free_flight_time(&mut rng, &info);
            }
        }
    }

    write_plots("poisson", "density", [plot_charge, plot_efield, plot_voltage]);
}
