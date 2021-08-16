#!/usr/bin/env python





import numpy as np

from gnome.utilities.weathering import (LeeHuibers,
                                        Riazi,
                                        Stokes,
                                        PiersonMoskowitz,
                                        DelvigneSweeney,
                                        DingFarmer,
                                        )


def test_lee_huibers():
    assert np.isclose(LeeHuibers.partition_coeff(92.1, 866.0), 1000)


def test_riazi():
    assert np.isclose(Riazi.mol_wt(300.0), 67.44, atol=0.001)
    assert np.isclose(Riazi.density(300.0), 669.43, atol=0.001)
    assert np.isclose(Riazi.molar_volume(300.0), 0.1, atol=0.001)

    assert np.isclose(Riazi.mol_wt(400.0), 113.756, atol=0.001)
    assert np.isclose(Riazi.density(400.0), 736.8, atol=0.001)
    assert np.isclose(Riazi.molar_volume(400.0), 0.154, atol=0.001)


def test_stokes():
    water_rho = 1000.0  # kg/m^3
    droplet_diameter = 0.0002  # 200 microns

    for oil_rho, expected in zip((900.0, 800.0, 700.0),
                                 (0.00217926, 0.00435852, 0.00653778)):
        delta_rho = water_rho - oil_rho
        assert np.isclose(Stokes.water_phase_xfer_velocity(delta_rho,
                                                           droplet_diameter),
                          expected)


def test_pierson_moskowitz():
    assert np.isclose(PiersonMoskowitz.significant_wave_height(10.0), 2.24337)
    assert np.isclose(PiersonMoskowitz.peak_wave_period(10.0), 7.5)


def test_delvigne_sweeney():
    wind_speed = 10.0
    T_w = PiersonMoskowitz.peak_wave_period(wind_speed)

    assert np.isclose(DelvigneSweeney.breaking_waves_frac(wind_speed, 10.0),
                      0.016)
    assert np.isclose(DelvigneSweeney.breaking_waves_frac(wind_speed, T_w),
                      0.0213333)


def test_ding_farmer():
    wind_speed = 10.0  # m/s
    rdelta = 200.0  # oil/water density difference (kg / m^3)
    droplet_diameter = 0.0002  # 200 microns

    wave_height = PiersonMoskowitz.significant_wave_height(wind_speed)
    wave_period = PiersonMoskowitz.peak_wave_period(wind_speed)

    f_bw = DelvigneSweeney.breaking_waves_frac(wind_speed, wave_period)
    k_w = Stokes.water_phase_xfer_velocity(rdelta, droplet_diameter)

    assert np.isclose(DingFarmer.calm_between_wave_breaks(0.5, 10),
                      15.0)
    assert np.isclose(DingFarmer.calm_between_wave_breaks(f_bw, wave_period),
                      347.8125)
    assert np.isclose(DingFarmer.refloat_time(wave_height, k_w),
                      386.0328)

    assert np.isclose(DingFarmer.water_column_time_fraction(f_bw,
                                                            wave_period,
                                                            wave_height,
                                                            k_w),
                      1.0)


def test_monohan():
    from gnome.utilities.weathering.monahan import Monahan

    assert np.isclose(Monahan.whitecap_decay_constant(0), 2.54)  # fresh water
    assert np.isclose(Monahan.whitecap_decay_constant(35), 3.85)  # salt water
